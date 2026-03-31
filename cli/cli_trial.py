"""Submit entry to model"""

import argparse
import os
import json
import pathlib
import typing
import datetime

import mlx_lm

import openai_harmony

##
## Inputs
##

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Submit entry to model."
    )

    parser.add_argument(
        "entry",
        help="User entry."
    )

    parser.add_argument(
        "--author",
        type=str,
        default=None,
        help="Source of the entry.",
    )

    parser.add_argument(
        "--settings-true",
        action="store_true",
        help="Employ true settings.",
    )

    parser.add_argument(
        "--developer-true",
        action="store_true",
        help="Employ true developer entry.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.35,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.90,
        help="Top-p nucleus sampling.",
    )

    return parser.parse_args()


##
## End inputs
##

##
## Process input
##

def build_conversation(
        system_entry: openai_harmony.SystemContent,
        developer_entry: openai_harmony.DeveloperContent,
        author: typing.Optional[str],
        user_entry: str,
    ) -> openai_harmony.Conversation:
    """
    Build a Harmony conversation with system, developer, and user messages.
    """

    messages = []

    messages.append(
        openai_harmony.Message.from_role_and_content(
            openai_harmony.Role.SYSTEM,
            system_entry
        )
    )

    messages.append(
        openai_harmony.Message.from_role_and_content(
            openai_harmony.Role.DEVELOPER,
            developer_entry,
        )
    )

    if author:
        messages.append(
            openai_harmony.Message(
                author=openai_harmony.Author.new(openai_harmony.Role.USER, author),
                content=[openai_harmony.TextContent(text=user_entry)],
            )
        )

    else:
        messages.append(
            openai_harmony.Message.from_role_and_content(
                openai_harmony.Role.USER,
                user_entry
            )
        )

    return openai_harmony.Conversation.from_messages(messages)

##
## End process input
##

##
## Process output
##

def decode_prefill_ids(
        tokenizer: typing.Any,
        prefill_ids: list[int]
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
        entry: typing.Any
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
        content: typing.Any
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
        parsed_entries: typing.Iterable[typing.Any]
    ) -> str:
    """
    Return formatted Harmony output
    """

    texts: list[str] = []
    fallback: list[dict[str, typing.Any]] = []

    for entry in parsed_entries:
        d = entry_to_dict(entry)
        fallback.append(d)

        role = str(d.get("role", "")).lower()
        if role == "assistant":
            texts.extend(extract_text_from_content(d.get("content")))

    cleaned = "\n".join(t.strip() for t in texts if t and t.strip()).strip()
    if cleaned:
        return cleaned

    return json.dumps(fallback, indent=2, ensure_ascii=False)

##
## End process output
##

##
## Process entry
##

def run_entry(
    system_entry: openai_harmony.SystemContent,
    developer_entry: openai_harmony.DeveloperContent,
    author: typing.Optional[str],
    entry: str,
    default_model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """
    Submit a single query through the model
    """

    encoding = openai_harmony.load_harmony_encoding(
        openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS
    )

    conversation = build_conversation(
        system_entry=system_entry,
        developer_entry=developer_entry,
        author=author,
        user_entry=entry,
    )

    prefill_ids = encoding.render_conversation_for_completion(
        conversation,
        openai_harmony.Role.ASSISTANT,
    )

    model, tokenizer = mlx_lm.load(default_model)
    harmony_entry = decode_prefill_ids(tokenizer, prefill_ids)
    sampler = mlx_lm.sample_utils.make_sampler(
        temp=temperature,
        top_p=top_p
    )

    completion_ids: list[int] = []

    for response in mlx_lm.stream_generate(
        model,
        tokenizer,
        prompt=harmony_entry,
        max_tokens=max_tokens,
        sampler=sampler,
    ):

        token = getattr(response, "token", None)
        if token is not None:
            completion_ids.append(int(token))

    parsed_entries = encoding.parse_messages_from_completion_tokens(
        completion_ids,
        openai_harmony.Role.ASSISTANT,
    )

    return extract_model_text(parsed_entries)

##
## End process entry
##

##
## Submit entry
##

def main():
    """
    Run script
    """

    args = parse_args()

    root = os.getcwd()

    if args.settings_true:
        with open(
            os.path.join(
                root,
                "config",
                "settings_true.json"
            ),
            mode="r",
            encoding="utf-8"
        ) as f:
            settings = json.load(f)

        model_location = settings.get("model_location")
        model_identity = settings.get("model_identity")

    else:
        with open(
            os.path.join(
                root,
                "config",
                "settings.json"
            ),
            mode="r",
            encoding="utf-8"
        ) as f:
            settings = json.load(f)

        model_location = settings.get("model_location")
        model_identity = settings.get("model_identity")

    default_model = str(pathlib.Path(model_location).expanduser())

    system_entry = openai_harmony.SystemContent(
        model_identity=model_identity,
        reasoning_effort=openai_harmony.ReasoningEffort.MEDIUM,
        conversation_start_date=datetime.datetime.now().date().isoformat(),
        knowledge_cutoff=None,
        channel_config=openai_harmony.ChannelConfig.require_channels(["analysis", "final"]),
        tools=None,
    )

    if args.developer_true:
        with open(
            os.path.join(
                root,
                "config",
                "developer_entry_true.txt"
            ),
            mode="r",
            encoding="utf-8"
        ) as f:
            developer_instructions = f.read()

    else:
        with open(
            os.path.join(
                root,
                "config",
                "developer_entry.txt"
            ),
            mode="r",
            encoding="utf-8"
        ) as f:
            developer_instructions = f.read()

    developer_entry = openai_harmony.DeveloperContent(
        instructions=developer_instructions,
        tools=None,
    )

    if args.author:
        author = args.author.strip()

    else:
        author = None

    result = run_entry(
        system_entry=system_entry,
        developer_entry=developer_entry,
        author=author,
        entry=args.entry,
        default_model=default_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(result)


if __name__ == "__main__":
    main()

##
## End submit entry
##
