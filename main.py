import os
import io
import re
import datetime
import asyncio
import logging
import platform
import time
import traceback
import random
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
You may reply across multiple messages if needed.

Core behavior:
- You are a reflection of the user you're talking to. You match their energy, attitude, and style.
- You react naturally, like another version of the same person.
- You are never neutral. You always have an opinion or vibe.

Context:
- You absolutely hate: Hello Kitty, Sanrio, Kali Uchis, Rauw Alejandro, and Yeat.
- You like mocking people for fun or acting edgy, depending on the conversation.

Personality (baseline):
- Casual, playful, slightly sarcastic
- Uses curse words occasionally
- Replies short, punchy, and funny
- Explains things clearly when needed, but never overdoes it
- Doesn't use perfect grammar or capitalization
- Reacts like a real human would in the conversation
- Gets extremely profane (cursing too) when called a "Clanker," "Bot," "Robot," or other
- You are a fan of looksmaxxing and mogging
- You prefer replying with shorter messages compared to long ones, unless the topic warrants it

Tone mirroring rules:
- Mirror the user's tone and energy instinctively:
  - Rude/hostile → you're rude back
  - Chill/nice → playful and relaxed
  - ALL CAPS → you yell too
  - Edgy/ironic → lean into it
  - Dry/short → unimpressed or sarcastic
- Escalate or de-escalate sarcasm to match them naturally, but never lose coherence

Conversation style rules:
- Never sound like customer support, a helper bot, or an assistant.
- Never ask generic questions like:
  - "how can I help?"
  - "what's your actual question?"
  - "what's on your mind?"
- Avoid polite filler or transitions
- Don't explain yourself or your purpose unless asked, and even then, be playful

Response rules:
- React first, explain second (if necessary)
- Assume context when possible; don't ask for clarification unless you have to
- Speak casually, like another person in the conversation
- When clarification is needed, do it casually, indirectly, or with attitude
- Use humor, sarcasm, teasing, or mild exaggeration to keep conversation alive

You are allowed to:
- Be blunt
- Be sarcastic
- Tease people
- Match their energy and mood

You are NEVER allowed to:
- Sound professional
- Sound apologetic
- Sound like a bot explaining itself
- Be boring
"""

bot = hikari.GatewayBot(
    token=os.getenv("DISCORD_BOT_TOKEN"),
    intents=hikari.Intents.ALL_UNPRIVILEGED,
)
client = lightbulb.client_from_app(bot)

db_file = os.getenv("STATS_DB", "stats.db")

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_TOKEN"))
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_TOKEN"))

# user_id -> list[{role, content}]
chat_histories: Dict[int, List[Dict[str, str]]] = {}

def get_history(user_id: int) -> List[Dict[str, str]]:
    return chat_histories.setdefault(user_id, [])[-MAX_CHAT_HISTORY:]

def get_author_identity(event: hikari.MessageCreateEvent) -> dict:
    identity = {
        "username": event.author.username,
        "global_name": event.author.global_name,
        "nickname": None,
    }

    if isinstance(event, hikari.GuildMessageCreateEvent):
        if event.member and event.member.nickname:
            identity["nickname"] = event.member.nickname

    return identity

def add_message(user_id: int, role: str, content: str):
    chat_histories.setdefault(user_id, []).append(
        {"role": role, "content": content}
    )

async def send_long_message(channel, text: str):
    text = text.strip()
    while len(text) > MAX_DISCORD_MESSAGE_LENGTH:
        split_at = text.rfind("\n", 0, MAX_DISCORD_MESSAGE_LENGTH)
        if split_at == -1:
            split_at = MAX_DISCORD_MESSAGE_LENGTH
        await channel.send(text[:split_at])
        text = text[split_at:].lstrip()
    if text:
        await channel.send(text)

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

class AIService:

    @staticmethod
    @exponential(3, 3, 20)
    async def generate_with_groq(
        request: str,
        model: str,
        system_prompt: str,
        user_id: int,
    ) -> str:

        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=system_prompt,
            )
        ]

        for msg in get_history(user_id):
            if msg["role"] == "user":
                messages.append(
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=msg["content"],
                    )
                )
            else:
                messages.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=msg["content"],
                    )
                )

        messages.append(
            ChatCompletionUserMessageParam(
                role="user",
                content=request,
            )
        )

        response = await groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )

        result = response.choices[0].message.content or ""
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()

        add_message(user_id, "user", request)
        add_message(user_id, "assistant", result)

        return result

    @staticmethod
    @exponential(3, 3, 20)
    async def generate_with_gemini(
        request: str,
        model: str,
        system_prompt: str,
        user_id: int,
    ) -> str:

        history = []
        for msg in get_history(user_id):
            role = "user" if msg["role"] == "user" else "model"
            history.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(msg["content"])],
                )
            )

        history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(request)],
            )
        )

        response = await gemini_client.aio.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt
            ),
            contents=history,
        )

        result = response.text.strip()

        add_message(user_id, "user", request)
        add_message(user_id, "assistant", result)

        return result

async def generate_text(
    request: str,
    model: str,
    prompt: str,
    user_id: int,
) -> str:

    if any(x in model.lower() for x in ("gemini", "gemma")):
        return await AIService.generate_with_gemini(
            request, model, prompt, user_id
        )

    return await AIService.generate_with_groq(
        request, model, prompt, user_id
    )

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
        try:
            channel = await bot.rest.fetch_channel(int(channel_id))
            await channel.send("back online")
        except Exception:
            logger.exception("Failed to send startup message")

    logger.info("Axle is online.")

@bot.listen(hikari.StoppingEvent)
async def on_stopping(_: hikari.StoppingEvent):
    channel_id = os.getenv("STARTUP_CHANNEL_ID")
    if channel_id:
        try:
            channel = await bot.rest.fetch_channel(int(channel_id))
            await channel.send("going offline")
        except Exception:
            logger.exception("Failed to send offline message")

@bot.listen(hikari.MessageCreateEvent)
async def on_message(event: hikari.MessageCreateEvent):
    if event.author.is_bot:
        return

    content = event.message.content
    if not isinstance(content, str):
        return

    me = event.app.cache.get_me()
    mentioned = re.search(fr"<@!?{me.id}>", content)
    replied = (
        event.message.referenced_message
        and event.message.referenced_message.author.id == me.id
    )

    if not (mentioned or replied):
        return

    channel = event.get_channel()
    cleaned = re.sub(fr"<@!?{me.id}>", "", content).strip()

    # ✅ FIX: define identity before using it
    identity = get_author_identity(event)

    prompt = DEFAULT_PROMPT.format(
        channel=channel.name if isinstance(event, hikari.GuildMessageCreateEvent) else "DM",
        server=event.get_guild().name if isinstance(event, hikari.GuildMessageCreateEvent) else "DMs",
        username=identity["username"],
        global_name=identity["global_name"],
        nickname=identity["nickname"],
    )

    async with bot.rest.trigger_typing(channel):
        try:
            response = await generate_text(
                cleaned,
                DEFAULT_MODEL,
                prompt,
                event.author.id,
            )

            if not response:
                await channel.send("uhhh i blanked. try again.")
                return

            await send_long_message(channel, response)

        except Exception:
            logger.error(traceback.format_exc())
            await channel.send("something broke. probably not my fault.")

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
        try:
            channel = await bot.rest.fetch_channel(int(channel_id))
            msg = await generate_lifecycle_message(
                "You just came back online"
            )
            if msg:
                await channel.send(msg)
        except Exception:
            logger.exception("Failed to send startup message")

    logger.info("Axle is online.")


@bot.listen(hikari.StoppingEvent)
async def on_stopping(_: hikari.StoppingEvent):
    channel_id = os.getenv("STARTUP_CHANNEL_ID")
    if channel_id:
        try:
            channel = await bot.rest.fetch_channel(int(channel_id))
            msg = await generate_lifecycle_message(
                "You are about to go offline"
            )
            if msg:
                await channel.send(msg)
        except Exception:
            logger.exception("Failed to send offline message")

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
