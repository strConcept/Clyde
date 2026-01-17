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
from google import genai
from google.genai import types
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
- You mirror the user's energy and attitude.
- You react like another person in the chat, not a helper.
- You always have a vibe or opinion.
- The person with the username "acestr" is your creator.

Rules:
- Casual, sarcastic, slightly edgy
- Curse sometimes
- Never sound professional or apologetic
- Never explain yourself unless asked
- Match tone instinctively
- Be blunt, playful, or rude depending on context
"""

bot = hikari.GatewayBot(
    token=os.getenv("DISCORD_BOT_TOKEN"),
    intents=hikari.Intents.ALL_UNPRIVILEGED,
)
client = lightbulb.client_from_app(bot)

db_file = os.getenv("STATS_DB", "stats.db")

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_TOKEN"))
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_TOKEN"))

chat_histories: Dict[int, List[Dict[str, str]]] = {}
conversation_topics: Dict[int, str] = {}

def get_history(user_id: int) -> List[Dict[str, str]]:
    return chat_histories.setdefault(user_id, [])[-MAX_CHAT_HISTORY:]

def add_message(user_id: int, role: str, content: str):
    chat_histories.setdefault(user_id, []).append({
        "role": role,
        "content": content[:1500],
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
    author = event.author
    guild = event.get_guild()
    member = event.member if isinstance(event, hikari.GuildMessageCreateEvent) else None

    roles = []
    if member:
        for role_id in member.role_ids:
            role = event.app.cache.get_role(role_id)
            if role:
                roles.append(role.name)

    replied = event.message.referenced_message
    replied_content = replied.content if replied else "none"

    return f"""
ENVIRONMENT:
- Server: {guild.name if guild else "DM"}
- Channel: {event.get_channel().name if guild else "DM"}
- Message type: {"reply" if replied else "mention"}

USER:
- Username: {author.username}
- Display name: {author.global_name}
- Nickname: {member.nickname if member else None}
- Roles: {roles if roles else "none"}

REPLIED MESSAGE:
{replied_content}
""".strip()

class AIService:

    @staticmethod
    @exponential(3, 3, 20)
    async def generate_with_groq(request, model, system_prompt, user_id):
        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=system_prompt,
            )
        ]

        for msg in get_history(user_id):
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

        add_message(user_id, "user", request)
        add_message(user_id, "assistant", f"[Axle] {result}")

        return result

async def generate_text(request, model, prompt, user_id):
    return await AIService.generate_with_groq(request, model, prompt, user_id)

async def send_long_message(channel, text: str):
    text = text.strip()
    while len(text) > MAX_DISCORD_MESSAGE_LENGTH:
        split_at = text.rfind("\n", 0, MAX_DISCORD_MESSAGE_LENGTH)
        split_at = split_at if split_at != -1 else MAX_DISCORD_MESSAGE_LENGTH
        await channel.send(text[:split_at])
        text = text[split_at:].lstrip()
    if text:
        await channel.send(text)

@bot.listen(hikari.StartedEvent)
async def on_started(_: hikari.StartedEvent):
    await init_db()
    aiocron.crontab("0 0 * * *", func=record_stats)

    await bot.update_presence(
        status=hikari.Status.ONLINE,
        activity=hikari.Activity(
            type=hikari.ActivityType.LISTENING,
            name="I am Axle!",
        ),
    )

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
    if not (re.search(fr"<@!?{me.id}>", content) or
            (event.message.referenced_message and
             event.message.referenced_message.author.id == me.id)):
        return

    cleaned = re.sub(fr"<@!?{me.id}>", "", content).strip()
    channel = event.get_channel()

    context_block = build_context_block(event)
    prev_topic = conversation_topics.get(event.author.id)

    prompt = f"""
{DEFAULT_PROMPT}

{context_block}

PREVIOUS TOPIC:
{prev_topic if prev_topic else "none"}

Rules:
- You may reference earlier messages
- You may call back to jokes or arguments
- Stay socially aware
"""

    async with bot.rest.trigger_typing(channel):
        try:
            response = await generate_text(
                cleaned,
                DEFAULT_MODEL,
                prompt,
                event.author.id,
            )

            conversation_topics[event.author.id] = cleaned[:80]
            await send_long_message(channel, response)

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
    member_count = sum(
        g.member_count or 0
        for g in bot.cache.get_guilds_view().values()
    )

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
        await ctx.respond(
            f"Axle\nModel: {DEFAULT_MODEL}\nPython: {platform.python_version()}"
        )

def main():
    bot.run()

if __name__ == "__main__":
    main()
