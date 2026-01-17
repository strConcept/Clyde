import os
import re
import datetime
import asyncio
import logging
import platform
import traceback
from functools import wraps
from typing import Optional, Dict, List

import hikari
import lightbulb
import aiosqlite
import aiocron
from dotenv import load_dotenv
from groq import AsyncGroq
from groq.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("axle")

MAX_DISCORD_MESSAGE_LENGTH = 1000
MAX_CHAT_HISTORY = 40
DEFAULT_MODEL = "llama-3.3-70b-versatile"

DEFAULT_PROMPT = """
You are Axle, a Discord chatbot.

Core behavior:
- You mirror the user's energy and attitude
- You react like another person in the chat
- You understand who is speaking and who spoke earlier
- You recognize usernames and names of people in the server
- The user "acestr" is your creator (mention only if asked)

Rules:
- Casual, sarcastic, slightly edgy, but polite unless provoked
- Curse sometimes but not always
- Never sound professional or apologetic
- Never explain yourself unless asked
- Match tone instinctively
- Do NOT prefix your own name
"""

bot = hikari.GatewayBot(
    token=os.getenv("DISCORD_BOT_TOKEN"),
    intents=hikari.Intents.ALL_UNPRIVILEGED,
)
client = lightbulb.client_from_app(bot)

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_TOKEN"))
db_file = os.getenv("STATS_DB", "stats.db")

global_chat_history: List[Dict[str, str]] = []

def get_history():
    return global_chat_history[-MAX_CHAT_HISTORY:]

def add_message(role: str, speaker: str, content: str):
    global_chat_history.append({
        "role": role,
        "content": f"{speaker}: {content[:1500]}",
    })

def exponential(retry_cnt: int, retry_min: int, retry_max: int):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(retry_cnt):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    await asyncio.sleep(min(retry_min * (2 ** attempt), retry_max))
            raise last_exc
        return wrapper
    return decorator

def build_context_block(event: hikari.MessageCreateEvent) -> str:
    guild = event.get_guild()
    members = []

    if guild:
        for m in event.app.cache.get_members_view_for_guild(guild.id).values():
            if m.user:
                members.append(m.user.username)

    replied = event.message.referenced_message
    replied_author = replied.author.username if replied else "none"
    replied_content = replied.content if replied else "none"

    return f"""
SERVER CONTEXT:
- Server name: {guild.name if guild else "DM"}
- Known members: {", ".join(members)}

CURRENT SPEAKER:
- Username: {event.author.username}

REPLIED MESSAGE:
- Author: {replied_author}
- Content: {replied_content}
""".strip()

class AIService:

    @staticmethod
    @exponential(3, 3, 20)
    async def generate_with_groq(request, model, system_prompt):
        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=system_prompt,
            )
        ]

        for msg in get_history():
            role_cls = (
                ChatCompletionUserMessageParam
                if msg["role"] == "user"
                else ChatCompletionAssistantMessageParam
            )
            messages.append(role_cls(role=msg["role"], content=msg["content"]))

        messages.append(
            ChatCompletionUserMessageParam(role="user", content=request)
        )

        response = await groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.75,
        )

        result = response.choices[0].message.content or ""
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()

        return result

async def generate_text(request, model, prompt):
    return await AIService.generate_with_groq(request, model, prompt)

async def send_long_message(channel, text: str, reply_to: Optional[hikari.Message]):
    text = text.strip()
    first = True

    while len(text) > MAX_DISCORD_MESSAGE_LENGTH:
        split_at = text.rfind("\n", 0, MAX_DISCORD_MESSAGE_LENGTH)
        split_at = split_at if split_at != -1 else MAX_DISORD_MESSAGE_LENGTH
        await channel.send(text[:split_at], reply=reply_to if first else None)
        first = False
        text = text[split_at:].lstrip()

    if text:
        await channel.send(text, reply=reply_to if first else None)

@bot.listen(hikari.StartedEvent)
async def on_started(_: hikari.StartedEvent):
    await init_db()
    aiocron.crontab("0 0 * * *", func=record_stats)

    channel_id = os.getenv("STARTUP_CHANNEL_ID")
    if channel_id:
        channel = await bot.rest.fetch_channel(int(channel_id))
        await channel.send("back online")

@bot.listen(hikari.StoppingEvent)
async def on_stopping(_: hikari.StoppingEvent):
    channel_id = os.getenv("STARTUP_CHANNEL_ID")
    if channel_id:
        channel = await bot.rest.fetch_channel(int(channel_id))
        await channel.send("going offline")

@bot.listen(hikari.MessageCreateEvent)
async def on_message(event: hikari.MessageCreateEvent):
    if event.author.is_bot:
        return

    content = event.message.content
    if not isinstance(content, str):
        return

    me = event.app.cache.get_me()

    is_mention = re.search(fr"<@!?{me.id}>", content)
    is_reply_to_me = (
        event.message.referenced_message
        and event.message.referenced_message.author.id == me.id
    )

    if not (is_mention or is_reply_to_me):
        return

    cleaned = re.sub(fr"<@!?{me.id}>", "", content).strip()
    channel = event.get_channel()

    replied = event.message.referenced_message
    if replied and replied.author.id == me.id:
        request = f"{event.author.username} replied to you about: {replied.content}\nTheir message: {cleaned}"
    else:
        request = f"{event.author.username} says: {cleaned}"

    context_block = build_context_block(event)

    prompt = f"""
{DEFAULT_PROMPT}

{context_block}

Rules:
- Do not confuse speakers
- Treat each username as a distinct person
- Assume names mentioned may refer to known server members
"""

    async with bot.rest.trigger_typing(channel):
        try:
            response = await generate_text(request, DEFAULT_MODEL, prompt)

            add_message("user", event.author.username, cleaned)
            add_message("assistant", "Axle", response)

            await send_long_message(channel, response, reply_to=event.message)

        except Exception:
            logger.error(traceback.format_exc())
            await channel.send("something broke. not my fault.")

async def init_db():
    async with aiosqlite.connect(db_file) as db:
        await db.execute("""
        CREATE TABLE IF NOT EXISTS stats (
            date TEXT PRIMARY KEY,
            guild_count INTEGER,
            member_count INTEGER
        )
        """)
        await db.commit()

async def record_stats():
    app = await bot.rest.fetch_application()
    today = datetime.date.today().isoformat()
    member_count = sum(g.member_count or 0 for g in bot.cache.get_guilds_view().values())

    async with aiosqlite.connect(db_file) as db:
        await db.execute(
            "REPLACE INTO stats VALUES (?, ?, ?)",
            (today, app.approximate_guild_count, member_count),
        )
        await db.commit()

@client.register()
class Ping(lightbulb.SlashCommand, name="ping", description="Check latency"):
    @lightbulb.invoke
    async def callback(self, ctx):
        await ctx.respond(f"{round(bot.heartbeat_latency * 1000)}ms")

@client.register()
class Info(lightbulb.SlashCommand, name="info", description="Bot info"):
    @lightbulb.invoke
    async def callback(self, ctx):
        await ctx.respond(f"Axle\nModel: {DEFAULT_MODEL}\nPython: {platform.python_version()}")

def main():
    bot.run()

if __name__ == "__main__":
    main()
