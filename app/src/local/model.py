"""Submit entry to model"""

import os
import typing

import openai_harmony
import openai

from app.src.helpers.variables import (
    ROOT,
    CUSTOM_DEVELOPER,
    SERVER_PATH,
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
        system_entry: str,
        developer_entry: str,
    ) -> list:
    """
    Build system and developer messages.
    """

    conversation.append(
        {
            "role": "system",
            "content": system_entry,
        }
    )

    conversation.append(
        {
            "role": "developer",
            "content": developer_entry,
        }
    )

    return conversation


def build_user_conversation(
        conversation: list,
        user_entry: str,
    ) -> list:
    """
    Build user messages.
    """

    if user_entry:
        conversation.append(
            {
                "role": "user",
                "content": user_entry,
            }
        )

    return conversation


def build_model_conversation(
        conversation: list,
        model_entry: str,
        model_channel: str,
    ) -> list:
    """
    Build model messages.
    """

    if model_channel in ("final"):
        conversation.append(
            {
                "role": "assistant",
                "content": model_entry,
            }
        )

    return conversation

##
## End process input
##

##
## Process output
##

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
    system_entry: str,
    developer_entry: str,
    memories_entry: dict,
    entries: list[dict],
) -> list:
    """
    Submit a single query through the model
    """

    conversation = []

    conversation = build_presets_conversation(
        conversation=conversation,
        system_entry=system_entry,
        developer_entry=developer_entry,
    )

    if memories_entry["content"]:
        entries = [memories_entry, *entries]

    for entry in entries:

        channel = entry.get("channel")
        content = entry.get("content")

        if not content:
            continue

        match channel:
            case "final":
                conversation = build_model_conversation(
                    conversation=conversation,
                    model_entry=content,
                    model_channel=channel,
            )
            case "user":
                conversation = build_user_conversation(
                    conversation=conversation,
                    user_entry=content,
            )

    client = openai.OpenAI(
        base_url=SERVER_PATH,
        api_key="o"
    )

    response = client.chat.completions.create(
        model=MODEL_PATH,
        messages=conversation,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=RETURN_TOKENS,
    )

    encoding = openai_harmony.load_harmony_encoding(
        openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS
    )

    response_tokens = encoding.encode(
        response.choices[0].message.content,
        allowed_special="all"
    )

    parsed_response = encoding.parse_messages_from_completion_tokens(
        response_tokens,
        role=openai_harmony.Role.ASSISTANT,
        strict=True,
    )

    return extract_model_text(parsed_response)

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

    result = run_entry(
        system_entry=MODEL_IDENTITY,
        memories_entry=memories_entry,
        developer_entry=developer_instructions,
        entries=entries,
    )

    return result

##
## End submit entry
##
