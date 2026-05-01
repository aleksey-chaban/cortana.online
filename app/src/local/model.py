"""Submit entry to model"""

import os
import typing
import datetime

import mlx_lm

import openai_harmony

from app.src.helpers.variables import (
    ROOT,
    CUSTOM_DEVELOPER,
    MODEL_PATH,
    MODEL_IDENTITY,
    RETURN_TOKENS,
    TEMPERATURE,
    TOP_P
)


##
## Process input
##

def build_presets_conversation(
        conversation: list,
        system_entry: openai_harmony.SystemContent,
        developer_entry: openai_harmony.DeveloperContent,
    ) -> list:
    """
    Build Harmony system and developer messages.
    """

    conversation.append(
        openai_harmony.Message.from_role_and_content(
            openai_harmony.Role.SYSTEM,
            system_entry,
        )
    )

    conversation.append(
        openai_harmony.Message.from_role_and_content(
            openai_harmony.Role.DEVELOPER,
            developer_entry,
        )
    )

    return conversation


def build_user_conversation(
        conversation: list,
        author: typing.Optional[str],
        user_entry: str,
    ) -> list:
    """
    Build Harmony user messages.
    """

    if user_entry:
        if author:
            conversation.append(
                openai_harmony.Message(
                    author=openai_harmony.Author.new(openai_harmony.Role.USER, author),
                    content=[openai_harmony.TextContent(text=user_entry)],
                )
            )

        else:
            conversation.append(
                openai_harmony.Message.from_role_and_content(
                    openai_harmony.Role.USER,
                    user_entry,
                )
            )

    return conversation


def build_model_conversation(
        conversation: list,
        model_name: str,
        model_entry: str,
        model_channel: str,
    ) -> list:
    """
    Build Harmony model messages.
    """

    if model_channel in ("analysis", "final"):
        conversation.append(
            openai_harmony.Message(
                author=openai_harmony.Author.new(openai_harmony.Role.ASSISTANT, model_name),
                channel=model_channel,
                content=[openai_harmony.TextContent(text=model_entry)],
            )
        )

    return conversation

##
## End process input
##

##
## Process output
##

def decode_prefill_ids(
        tokenizer: typing.Any,
        prefill_ids: list[int],
    ) -> str:
    """
    Decode the Harmony token sequence back into text while preserving special tokens.
    """

    decode_kwargs = {
        "skip_special_tokens": False,
        "clean_up_tokenization_spaces": False,
    }

    try:
        return tokenizer.decode(prefill_ids, **decode_kwargs)

    except TypeError:
        decode_kwargs.pop("clean_up_tokenization_spaces", None)
        return tokenizer.decode(prefill_ids, **decode_kwargs)


def entry_to_dict(
        entry: typing.Any,
    ) -> dict[str, typing.Any]:
    """
    Convert Harmony response to dictionary format
    """

    if hasattr(entry, "to_dict"):
        return entry.to_dict()

    if isinstance(entry, dict):
        return entry

    return {"value": str(entry)}


def extract_text_from_content(
        content: typing.Any,
    ) -> list[str]:
    """
    Prepare Harmony output for formatting
    """

    out: list[str] = []

    if content is None:
        return out

    if isinstance(content, str):
        return [content]

    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            val = content.get(key)
            if isinstance(val, str) and val.strip():
                out.append(val)
        return out

    if isinstance(content, list):
        for item in content:
            out.extend(extract_text_from_content(item))
        return out

    text = str(content).strip()
    if text:
        out.append(text)
    return out


def extract_model_text(
        parsed_entries: typing.Iterable[typing.Any],
    ) -> list:
    """
    Return formatted Harmony output
    """

    texts: list[str] = []
    output_list: list[dict[str, typing.Any]] = []

    for entry in parsed_entries:
        d = entry_to_dict(entry)
        output_list.append(d)

        role = str(d.get("role", "")).lower()
        if role == "assistant":
            texts.extend(extract_text_from_content(d.get("content")))

    return output_list

##
## End process output
##

##
## Process entry
##

def run_entry(
    system_entry: openai_harmony.SystemContent,
    developer_entry: openai_harmony.DeveloperContent,
    memories_entry: dict,
    entries: list[dict],
) -> list:
    """
    Submit a single query through the model
    """

    encoding = openai_harmony.load_harmony_encoding(
        openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS
    )

    conversation = []

    conversation = build_presets_conversation(
        conversation=conversation,
        system_entry=system_entry,
        developer_entry=developer_entry,
    )

    if memories_entry["content"]:
        entries = [memories_entry, *entries]

    for entry in entries:

        author = entry.get("author")
        channel = entry.get("channel")
        content = entry.get("content")

        if not content:
            continue

        match channel:
            case "analysis" | "final":
                conversation = build_model_conversation(
                    conversation=conversation,
                    model_name=MODEL_PATH,
                    model_entry=content,
                    model_channel=channel,
            )
            case "user":
                conversation = build_user_conversation(
                    conversation=conversation,
                    author=author,
                    user_entry=content,
            )

    conversation = openai_harmony.Conversation.from_messages(conversation)

    prefill_ids = encoding.render_conversation_for_completion(
        conversation,
        openai_harmony.Role.ASSISTANT,
    )

    token_count = len(prefill_ids)

    model, tokenizer = mlx_lm.load(MODEL_PATH)
    harmony_entry = decode_prefill_ids(tokenizer, prefill_ids)
    sampler = mlx_lm.sample_utils.make_sampler(
        temp=TEMPERATURE,
        top_p=TOP_P,
    )

    completion_ids: list[int] = []

    for response in mlx_lm.stream_generate(
        model,
        tokenizer,
        prompt=harmony_entry,
        max_tokens=RETURN_TOKENS,
        sampler=sampler,
    ):

        token = getattr(response, "token", None)
        if token is not None:
            completion_ids.append(int(token))

    parsed_entries = encoding.parse_messages_from_completion_tokens(
        completion_ids,
        openai_harmony.Role.ASSISTANT,
    )

    return [
        token_count,
        extract_model_text(parsed_entries)
    ]

##
## End process entry
##

##
## Submit entry
##

def submit_entry(
    entries: list[dict],
    memories_entry: dict,
) -> list:
    """
    Submit entry to the model
    """

    system_entry = openai_harmony.SystemContent(
        model_identity=MODEL_IDENTITY,
        reasoning_effort=openai_harmony.ReasoningEffort.LOW,
        conversation_start_date=datetime.datetime.now().date().isoformat(),
        knowledge_cutoff=None,
        channel_config=openai_harmony.ChannelConfig.require_channels(["analysis", "final"]),
        tools=None,
    )

    with open(
        os.path.join(
            ROOT,
            "config",
            "developer_entry.txt"
        ),
        mode="r",
        encoding="utf-8"
    ) as f:
        developer_instructions = f.read()

    if CUSTOM_DEVELOPER:
        with open(
            os.path.join(
                ROOT,
                "config",
                "developer_entry_true.txt"
            ),
            mode="r",
            encoding="utf-8"
        ) as f:
            developer_instructions = f.read()

    developer_entry = openai_harmony.DeveloperContent(
        instructions=developer_instructions,
        tools=None,
    )

    token_count, result = run_entry(
        system_entry=system_entry,
        memories_entry=memories_entry,
        developer_entry=developer_entry,
        entries=entries,
    )

    return [
        token_count,
        result
    ]

##
## End submit entry
##
