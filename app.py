import base64
import json
import os
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "few_shot_examples.json"
SYSTEM_PROMPT_FILE = BASE_DIR / "prompts" / "system_prompt.txt"
USER_PROMPT_FILE = BASE_DIR / "prompts" / "user_prompt_template.txt"


# ---------------------------
# File / text helpers
# ---------------------------
def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_examples() -> list:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_multiline(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def list_to_bullets(items: list[str]) -> str:
    if not items:
        return "- None provided"
    return "\n".join(f"- {item}" for item in items)


def clean_field(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "n/a", "na", "no", "nope"}:
        return ""
    return text


# ---------------------------
# Few-shot helpers
# ---------------------------
def format_example(example: dict, index: int) -> str:
    market = example.get("market", {})
    summary = example.get("property_summary", {})
    output = example.get("output", {})

    features = ", ".join(summary.get("features", []))

    return f"""
Example {index}

MARKET:
- City: {market.get("city", "")}
- State: {market.get("state", "")}
- County: {market.get("county", "")}

INPUT:
- Price: {summary.get("price", "")}
- Beds: {summary.get("beds", "")}
- Baths: {summary.get("baths", "")}
- Square Feet: {summary.get("sqft", "")}
- Property Subtype: {summary.get("property_subtype", "")}
- Features: {features}

OUTPUT:
headline: {output.get("headline", "")}
mls_description: {output.get("mls_description", "")}
""".strip()


def select_examples(all_examples: list, city: str, max_examples: int = 3) -> list:
    city = (city or "").strip().lower()

    same_city = [
        ex for ex in all_examples
        if ex.get("market", {}).get("city", "").strip().lower() == city
    ]
    other = [
        ex for ex in all_examples
        if ex.get("market", {}).get("city", "").strip().lower() != city
    ]

    selected = same_city[:max_examples]
    if len(selected) < max_examples:
        selected.extend(other[: max_examples - len(selected)])

    return selected[:max_examples]


def build_few_shot_block(examples: list) -> str:
    return "\n\n".join(format_example(ex, i + 1) for i, ex in enumerate(examples))


# ---------------------------
# Prompt builders
# ---------------------------
def build_user_prompt(property_data: dict, examples: list) -> str:
    template = load_text_file(USER_PROMPT_FILE)
    few_shot_examples = build_few_shot_block(examples)

    return template.format(
        address=clean_field(property_data.get("address", "")),
        city=clean_field(property_data.get("city", "")),
        state=clean_field(property_data.get("state", "")),
        price=clean_field(property_data.get("price", "")),
        beds=property_data.get("beds", ""),
        baths=property_data.get("baths", ""),
        sqft=property_data.get("sqft", ""),
        year_built=clean_field(property_data.get("year_built", "")),
        property_subtype=clean_field(property_data.get("property_subtype", "")),
        lot_size=clean_field(property_data.get("lot_size", "")),
        garage=clean_field(property_data.get("garage", "")),
        basement=clean_field(property_data.get("basement", "")),
        shop_outbuildings=clean_field(property_data.get("shop_outbuildings", "")),
        interior_features=list_to_bullets(property_data.get("interior_features", [])),
        exterior_features=list_to_bullets(property_data.get("exterior_features", [])),
        recent_updates=list_to_bullets(property_data.get("recent_updates", [])),
        location_highlights=list_to_bullets(property_data.get("location_highlights", [])),
        tone=property_data.get("tone", "professional"),
        length=property_data.get("length", "medium"),
        few_shot_examples=few_shot_examples,
    )


def build_listing_prompt(property_data: dict, examples: list) -> str:
    system_prompt = load_text_file(SYSTEM_PROMPT_FILE)
    user_prompt = build_user_prompt(property_data, examples)

    return f"""{system_prompt}

{user_prompt}

Additional rules:
- Do not mention the absence of a feature unless it is clearly beneficial and explicitly relevant.
- Do not mention "no basement" unless the user specifically wants that included.
- The social_caption should be more engaging than the listing description, not just a shorter summary.
- The social_caption should emphasize the strongest selling points in a more attention-grabbing but still professional way.

Return valid JSON with exactly these keys:
headline
mls_description
portal_description
social_caption

Requirements for social_caption:
- 2 to 4 short sentences max
- More engaging than the listing description
- Focus on the strongest selling points
- Sound appropriate for Facebook or Instagram
- Include a light call to action
- The final sentence should feel slightly more inviting and energetic than the MLS copy
- A subtle exclamation point is acceptable in the final sentence when natural
- Do not simply summarize the MLS description
- Avoid cheesy clichés or overly exaggerated language
"""


def build_marketing_prompt(property_data: dict, examples: list) -> str:
    few_shot_examples = build_few_shot_block(examples)

    return f"""
You are an AI marketing assistant for a residential real estate agent serving Fremont County, Wyoming, especially Riverton and Lander.

Your job is to create a cohesive marketing package for a home listing using only the facts provided.

Rules:
- Use only the facts provided.
- Do not invent features, views, neighborhood benefits, acreage details, or utilities.
- Match the tone of a strong local realtor in Fremont County.
- Keep the writing marketable, natural, and specific.
- Avoid cheesy clichés like "dream home," "won't last long," or exaggerated fluff.
- Do not mention the absence of features unless their absence is clearly beneficial and explicitly relevant.
- Marketing copy should be more engaging than MLS copy, but still credible and professional.

CURRENT PROPERTY

Address: {clean_field(property_data.get("address", ""))}
City: {clean_field(property_data.get("city", ""))}
State: {clean_field(property_data.get("state", ""))}
Price: {clean_field(property_data.get("price", ""))}
Beds: {property_data.get("beds", "")}
Baths: {property_data.get("baths", "")}
Square Feet: {property_data.get("sqft", "")}
Year Built: {clean_field(property_data.get("year_built", ""))}
Property Subtype: {clean_field(property_data.get("property_subtype", ""))}
Lot Size: {clean_field(property_data.get("lot_size", ""))}
Garage: {clean_field(property_data.get("garage", ""))}
Basement: {clean_field(property_data.get("basement", ""))}
Shop / Outbuildings: {clean_field(property_data.get("shop_outbuildings", ""))}

Interior Features:
{list_to_bullets(property_data.get("interior_features", []))}

Exterior Features:
{list_to_bullets(property_data.get("exterior_features", []))}

Recent Updates:
{list_to_bullets(property_data.get("recent_updates", []))}

Location Highlights:
{list_to_bullets(property_data.get("location_highlights", []))}

Tone: {property_data.get("tone", "professional")}
Length: {property_data.get("length", "medium")}

STYLE EXAMPLES

{few_shot_examples}

Return valid JSON with exactly these keys:
facebook_post
instagram_caption
buyer_email
sms_text
property_highlights
open_house_post

Requirements:
- facebook_post: 3 to 5 sentences, engaging and informative, suitable for a Facebook listing post
- instagram_caption: 2 to 4 short sentences, more punchy and scroll-stopping, still professional
- buyer_email: short email-style property announcement for interested buyers
- sms_text: short text message blast, concise and natural
- property_highlights: an array of 5 to 8 bullet-style highlight strings
- open_house_post: a promotional post for an open house, even if no exact date/time is given; if not provided, write with placeholders like "Join us this weekend" instead of inventing specifics
"""


def build_rewrite_listing_prompt(raw_text: str, city: str, tone: str, length: str, examples: list) -> str:
    few_shot_examples = build_few_shot_block(examples)

    return f"""
You are rewriting and improving an existing residential real estate listing or raw property notes.

Target market:
- Fremont County, Wyoming
- Especially Riverton and Lander

Instructions:
- Use the pasted content only as source material.
- Do not invent facts.
- Keep the writing specific to the property.
- Match the tone, pacing, and practical style of a local Fremont County realtor.
- Avoid generic real estate fluff and fair-housing-risky language.
- Clean up awkward wording, repetition, and weak structure.
- If details are rough notes instead of full sentences, convert them into polished marketing copy.
- Do not mention the absence of features unless their absence is clearly beneficial and explicitly relevant.
- Do not mention "no basement" unless the user specifically wants that included.

Requested city context: {city}
Requested tone: {tone}
Requested length: {length}

STYLE EXAMPLES

{few_shot_examples}

PASTED SOURCE MATERIAL

{raw_text}

Return valid JSON with exactly these keys:
headline
mls_description
portal_description
social_caption

Requirements for social_caption:
- 2 to 4 short sentences max
- More engaging than the listing description
- Focus on the strongest selling points
- Sound appropriate for Facebook or Instagram
- Include a light call to action
- The final sentence should feel slightly more inviting and energetic than the MLS copy
- A subtle exclamation point is acceptable in the final sentence when natural
- Do not simply summarize the MLS description
- Avoid cheesy clichés or overly exaggerated language
"""


def build_rewrite_marketing_prompt(raw_text: str, city: str, tone: str, length: str, examples: list) -> str:
    few_shot_examples = build_few_shot_block(examples)

    return f"""
You are rewriting rough listing notes or pasted property text into a complete residential real estate marketing package.

Target market:
- Fremont County, Wyoming
- Especially Riverton and Lander

Instructions:
- Use the pasted content only as source material.
- Do not invent facts.
- Turn rough notes into polished, marketable output.
- Keep the tone professional, local, and credible.
- Avoid generic fluff and cheesy clichés.

Requested city context: {city}
Requested tone: {tone}
Requested length: {length}

STYLE EXAMPLES

{few_shot_examples}

PASTED SOURCE MATERIAL

{raw_text}

Return valid JSON with exactly these keys:
facebook_post
instagram_caption
buyer_email
sms_text
property_highlights
open_house_post

Requirements:
- facebook_post: 3 to 5 sentences, engaging and informative
- instagram_caption: 2 to 4 short sentences, punchy but professional
- buyer_email: short email announcement
- sms_text: concise text blast
- property_highlights: an array of 5 to 8 bullet-style highlight strings
- open_house_post: promotional open house copy using placeholders if date/time are not provided
"""


def build_weekly_content_prompt(content_type: str, area_focus: str, tone: str, extra_notes: str, include_graphic: bool) -> str:
    graphic_instruction = """
Also return:
graphic_headline
graphic_prompt

Graphic rules:
- The image must look clean, polished, branded, and usable for a real estate social post.
- Avoid photorealistic fake people, fake houses, cheesy sales imagery, cluttered layouts, and generic AI slop.
- Favor high-quality editorial social graphics, modern infographic cards, minimal branded layouts, or clean educational post visuals.
- Keep the design visually strong, readable, and social-media friendly.
- If text appears in the image, keep it short and bold.
- The prompt should be ready for direct image generation.
""" if include_graphic else """
Also return:
graphic_headline
graphic_prompt

Set graphic_headline and graphic_prompt to empty strings.
"""

    return f"""
You are an AI marketing assistant for a residential real estate team serving Riverton, Lander, and Fremont County, Wyoming.

Create a weekly marketing content package that helps the agent stay visible, useful, and credible online.

Content type requested: {content_type}
Area focus: {area_focus}
Tone: {tone}

Extra notes:
{extra_notes if extra_notes.strip() else "None provided"}

Requirements:
- Make the content useful, relevant, and engaging.
- Avoid clichés, fluff, and generic corporate language.
- Keep the tone natural and local.
- Do not invent hard market statistics unless they are explicitly provided in the notes.
- If the topic is a market update, keep it general unless specific numbers are provided.
- The content should sound like a real local agent, not a motivational influencer.

Return valid JSON with exactly these keys:
topic
facebook_post
instagram_caption
short_video_script
graphic_headline
graphic_prompt

Field requirements:
- topic: a clear weekly topic title
- facebook_post: 3 to 5 sentences
- instagram_caption: 2 to 4 short sentences
- short_video_script: 4 short sections labeled Hook, Talking Points, CTA
{graphic_instruction}
"""


# ---------------------------
# API helpers
# ---------------------------
def generate_json(prompt: str) -> dict:
    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )

    text = response.output_text.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    return json.loads(text)


def generate_image_bytes(prompt: str, size: str = "1024x1024") -> bytes:
    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=size,
        output_format="png",
    )
    return base64.b64decode(result.data[0].b64_json)


# ---------------------------
# Quick paste parser helpers
# ---------------------------
def extract_labeled_value(text: str, labels: list[str]) -> str:
    lines = text.splitlines()
    for line in lines:
        for label in labels:
            pattern = rf"^\s*{re.escape(label)}\s*:\s*(.+?)\s*$"
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).strip()
    return ""


def extract_section_items(text: str, section_name: str) -> list[str]:
    lines = text.splitlines()
    items = []
    in_section = False

    for line in lines:
        stripped = line.strip()

        if re.match(rf"^{re.escape(section_name)}\s*:\s*$", stripped, re.IGNORECASE):
            in_section = True
            continue

        if in_section and re.match(r"^[A-Za-z /&]+:\s*$", stripped):
            break

        if in_section:
            cleaned = stripped.lstrip("-•").strip()
            if cleaned:
                items.append(cleaned)

    return items


def parse_property_block(raw_text: str) -> dict:
    return {
        "address": extract_labeled_value(raw_text, ["Address"]),
        "city": extract_labeled_value(raw_text, ["City"]),
        "state": extract_labeled_value(raw_text, ["State"]),
        "price": extract_labeled_value(raw_text, ["Price"]),
        "beds": extract_labeled_value(raw_text, ["Beds", "Bedrooms"]),
        "baths": extract_labeled_value(raw_text, ["Baths", "Bathrooms"]),
        "sqft": extract_labeled_value(raw_text, ["Sqft", "Square Feet"]),
        "year_built": extract_labeled_value(raw_text, ["Year Built"]),
        "property_subtype": extract_labeled_value(raw_text, ["Property Subtype", "Subtype"]),
        "lot_size": extract_labeled_value(raw_text, ["Lot Size", "Lot Size / Acres", "Acres"]),
        "garage": extract_labeled_value(raw_text, ["Garage"]),
        "basement": extract_labeled_value(raw_text, ["Basement"]),
        "shop_outbuildings": extract_labeled_value(raw_text, ["Shop / Outbuildings", "Shop", "Outbuildings"]),
        "interior_features": extract_section_items(raw_text, "Interior Features"),
        "exterior_features": extract_section_items(raw_text, "Exterior Features"),
        "recent_updates": extract_section_items(raw_text, "Recent Updates"),
        "location_highlights": extract_section_items(raw_text, "Location Highlights"),
    }


def safe_int(value: str, default: int = 0) -> int:
    try:
        return int(float(str(value).replace(",", "").strip()))
    except Exception:
        return default


def safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return default


# ---------------------------
# Output helpers
# ---------------------------
def render_copy_button(label: str, text: str, button_id: str):
    safe_text = json.dumps(text)
    components.html(
        f"""
        <div style="margin-top:2px; margin-bottom:2px;">
            <button
                onclick='navigator.clipboard.writeText({safe_text}).then(() => {{
                    const btn = document.getElementById("{button_id}");
                    if (btn) {{
                        const original = btn.innerText;
                        btn.innerText = "Copied!";
                        setTimeout(() => btn.innerText = original, 1200);
                    }}
                }})'
                id="{button_id}"
                style="
                    background-color:#f0f2f6;
                    border:1px solid #d0d7de;
                    border-radius:8px;
                    padding:6px 12px;
                    cursor:pointer;
                    font-size:14px;
                "
            >
                {label}
            </button>
        </div>
        """,
        height=42,
    )


def render_output_block(title: str, text: str, key_prefix: str, filename: str, height: int = 180):
    st.subheader(title)
    st.text_area(f"{title} Output", value=text, height=height, key=f"{key_prefix}_text")
    col1, col2 = st.columns([1, 1])
    with col1:
        render_copy_button(f"Copy {title}", text, f"{key_prefix}_copy_btn")
    with col2:
        st.download_button(
            label=f"Download {title}",
            data=text,
            file_name=filename,
            mime="text/plain",
            key=f"{key_prefix}_download"
        )


def render_highlights_block(highlights: list, key_prefix: str):
    text = "\n".join(f"• {item}" for item in highlights)
    render_output_block("Property Highlights", text, key_prefix, "property_highlights.txt", height=180)


def build_combined_export(listing_output: dict | None = None, marketing_output: dict | None = None, weekly_output: dict | None = None) -> str:
    sections = []

    if listing_output:
        sections.append(
            f"""HEADLINE
{listing_output.get("headline", "")}

MLS DESCRIPTION
{listing_output.get("mls_description", "")}

PORTAL DESCRIPTION
{listing_output.get("portal_description", "")}

SOCIAL CAPTION
{listing_output.get("social_caption", "")}
"""
        )

    if marketing_output:
        highlights = marketing_output.get("property_highlights", [])
        highlights_text = "\n".join(f"• {item}" for item in highlights)

        sections.append(
            f"""FACEBOOK POST
{marketing_output.get("facebook_post", "")}

INSTAGRAM CAPTION
{marketing_output.get("instagram_caption", "")}

BUYER EMAIL
{marketing_output.get("buyer_email", "")}

SMS TEXT
{marketing_output.get("sms_text", "")}

PROPERTY HIGHLIGHTS
{highlights_text}

OPEN HOUSE POST
{marketing_output.get("open_house_post", "")}
"""
        )

    if weekly_output:
        sections.append(
            f"""WEEKLY TOPIC
{weekly_output.get("topic", "")}

FACEBOOK POST
{weekly_output.get("facebook_post", "")}

INSTAGRAM CAPTION
{weekly_output.get("instagram_caption", "")}

SHORT VIDEO SCRIPT
{weekly_output.get("short_video_script", "")}

GRAPHIC HEADLINE
{weekly_output.get("graphic_headline", "")}
"""
        )

    return "\n\n".join(sections)


def reset_structured_form():
    st.session_state.address = ""
    st.session_state.city = "Riverton"
    st.session_state.state = "WY"
    st.session_state.price = ""
    st.session_state.beds = 0
    st.session_state.baths = 0.0
    st.session_state.sqft = 0
    st.session_state.year_built = ""
    st.session_state.property_subtype = "single-family"
    st.session_state.lot_size = ""
    st.session_state.garage = ""
    st.session_state.basement = ""
    st.session_state.shop_outbuildings = ""
    st.session_state.interior_features = ""
    st.session_state.exterior_features = ""
    st.session_state.recent_updates = ""
    st.session_state.location_highlights = ""


# ---------------------------
# Session state defaults
# ---------------------------
default_fields = {
    "address": "",
    "city": "Riverton",
    "state": "WY",
    "price": "",
    "beds": 0,
    "baths": 0.0,
    "sqft": 0,
    "year_built": "",
    "property_subtype": "single-family",
    "lot_size": "",
    "garage": "",
    "basement": "",
    "shop_outbuildings": "",
    "interior_features": "",
    "exterior_features": "",
    "recent_updates": "",
    "location_highlights": "",
}

for key, value in default_fields.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Fremont County Residential AI", page_icon="🏡")
st.title("Fremont County Residential AI")
st.caption("Listing + marketing + weekly content tools for Riverton / Lander / Fremont County.")

page_mode = st.radio(
    "Choose mode",
    ["Structured Form", "Paste Existing Listing / Notes", "Weekly Marketing Content"],
    horizontal=True
)

if page_mode == "Structured Form":
    st.subheader("Quick Paste Property Details")
    quick_paste = st.text_area(
        "Paste labeled property details here to auto-fill the form",
        height=220,
        placeholder="""Address: 123 Main St
City: Riverton
State: WY
Price: $329,000
Beds: 3
Baths: 2
Sqft: 1780
Year Built: 2006
Property Subtype: single-family
Lot Size / Acres: 0.24 acres
Garage: Attached 2 car
Basement: None
Shop / Outbuildings: Storage shed

Interior Features:
- Open concept living area
- Large kitchen island
- Walk-in closet
- Stainless steel appliances

Exterior Features:
- Fenced backyard
- Covered patio
- Automatic sprinklers

Recent Updates:
- New roof in 2023
- New water heater

Location Highlights:
- Quiet street
- Close to schools
- Convenient access to shopping"""
    )

    col_parse, col_clear = st.columns([1, 1])
    with col_parse:
        if st.button("Parse Details Into Form"):
            parsed = parse_property_block(quick_paste)

            if parsed["address"]:
                st.session_state.address = parsed["address"]

            parsed_city = parsed["city"].strip()
            if parsed_city in ["Riverton", "Lander", "Other"]:
                st.session_state.city = parsed_city
            elif parsed_city:
                st.session_state.city = "Other"

            if parsed["state"]:
                st.session_state.state = parsed["state"]
            if parsed["price"]:
                st.session_state.price = parsed["price"]

            st.session_state.beds = safe_int(parsed["beds"], 0)
            st.session_state.baths = safe_float(parsed["baths"], 0.0)
            st.session_state.sqft = safe_int(parsed["sqft"], 0)

            if parsed["year_built"]:
                st.session_state.year_built = parsed["year_built"]

            normalized_subtype = parsed["property_subtype"].strip().lower()
            allowed_subtypes = [
                "single-family", "rural residential", "manufactured on foundation", "townhome", "condo", "other"
            ]
            if normalized_subtype in allowed_subtypes:
                st.session_state.property_subtype = normalized_subtype

            if parsed["lot_size"]:
                st.session_state.lot_size = parsed["lot_size"]
            if parsed["garage"]:
                st.session_state.garage = parsed["garage"]
            if parsed["basement"]:
                st.session_state.basement = parsed["basement"]
            if parsed["shop_outbuildings"]:
                st.session_state.shop_outbuildings = parsed["shop_outbuildings"]

            st.session_state.interior_features = "\n".join(parsed["interior_features"])
            st.session_state.exterior_features = "\n".join(parsed["exterior_features"])
            st.session_state.recent_updates = "\n".join(parsed["recent_updates"])
            st.session_state.location_highlights = "\n".join(parsed["location_highlights"])

            st.success("Parsed property details into the form.")

    with col_clear:
        if st.button("Clear Form"):
            reset_structured_form()
            st.success("Form cleared.")

    with st.form("listing_form"):
        col1, col2 = st.columns(2)

        with col1:
            address = st.text_input("Address", key="address")
            city = st.selectbox("City", ["Riverton", "Lander", "Other"], key="city")
            state = st.text_input("State", key="state")
            price = st.text_input("Price", placeholder="e.g. $335,000", key="price")
            beds = st.number_input("Beds", min_value=0, step=1, key="beds")
            baths = st.number_input("Baths", min_value=0.0, step=0.25, key="baths")
            sqft = st.number_input("Square Feet", min_value=0, step=1, key="sqft")
            year_built = st.text_input("Year Built", key="year_built")
            property_subtype = st.selectbox(
                "Property Subtype",
                ["single-family", "rural residential", "manufactured on foundation", "townhome", "condo", "other"],
                key="property_subtype"
            )

        with col2:
            lot_size = st.text_input("Lot Size / Acres", key="lot_size")
            garage = st.text_input("Garage", key="garage")
            basement = st.text_input("Basement", key="basement")
            shop_outbuildings = st.text_input("Shop / Outbuildings", key="shop_outbuildings")
            tone = st.selectbox("Tone", ["professional", "warm", "upscale", "concise"])
            length = st.selectbox("Length", ["short", "medium", "long"])

        interior_features = st.text_area("Interior Features (one per line)", key="interior_features")
        exterior_features = st.text_area("Exterior Features (one per line)", key="exterior_features")
        recent_updates = st.text_area("Recent Updates (one per line)", key="recent_updates")
        location_highlights = st.text_area("Location Highlights (one per line)", key="location_highlights")

        st.subheader("Choose what to generate")
        gen_col1, gen_col2, gen_col3 = st.columns(3)
        with gen_col1:
            generate_listing = st.form_submit_button("Generate Listing Copy")
        with gen_col2:
            generate_marketing = st.form_submit_button("Generate Marketing Package")
        with gen_col3:
            generate_both = st.form_submit_button("Generate Both")

    if generate_listing or generate_marketing or generate_both:
        try:
            if not api_key:
                st.error("Missing OPENAI_API_KEY in environment or deployment secrets.")
                st.stop()

            all_examples = load_examples()
            selected_examples = select_examples(all_examples, city=city, max_examples=3)

            property_data = {
                "address": address,
                "city": city,
                "state": state,
                "price": price,
                "beds": beds,
                "baths": baths,
                "sqft": sqft,
                "year_built": year_built,
                "property_subtype": property_subtype,
                "lot_size": lot_size,
                "garage": garage,
                "basement": basement,
                "shop_outbuildings": shop_outbuildings,
                "interior_features": parse_multiline(interior_features),
                "exterior_features": parse_multiline(exterior_features),
                "recent_updates": parse_multiline(recent_updates),
                "location_highlights": parse_multiline(location_highlights),
                "tone": tone,
                "length": length,
            }

            listing_output = None
            marketing_output = None

            if generate_listing or generate_both:
                with st.spinner("Generating listing copy..."):
                    listing_prompt = build_listing_prompt(property_data, selected_examples)
                    listing_output = generate_json(listing_prompt)

            if generate_marketing or generate_both:
                with st.spinner("Generating marketing package..."):
                    marketing_prompt = build_marketing_prompt(property_data, selected_examples)
                    marketing_output = generate_json(marketing_prompt)

            st.success("Done.")

            if listing_output:
                st.header("Listing Copy")
                render_output_block("Headline", listing_output["headline"], "headline", "headline.txt", height=80)
                render_output_block("MLS Description", listing_output["mls_description"], "mls", "mls_description.txt", height=220)
                render_output_block("Portal Description", listing_output["portal_description"], "portal", "portal_description.txt", height=220)
                render_output_block("Social Caption", listing_output["social_caption"], "social", "social_caption.txt", height=140)

            if marketing_output:
                st.header("Marketing Package")
                render_output_block("Facebook Post", marketing_output["facebook_post"], "facebook", "facebook_post.txt", height=180)
                render_output_block("Instagram Caption", marketing_output["instagram_caption"], "instagram", "instagram_caption.txt", height=160)
                render_output_block("Buyer Email", marketing_output["buyer_email"], "buyer_email", "buyer_email.txt", height=220)
                render_output_block("SMS Text", marketing_output["sms_text"], "sms_text", "sms_text.txt", height=120)
                render_highlights_block(marketing_output["property_highlights"], "property_highlights")
                render_output_block("Open House Post", marketing_output["open_house_post"], "open_house_post", "open_house_post.txt", height=180)

            combined_export = build_combined_export(listing_output, marketing_output, None)
            st.header("Export All")
            st.download_button(
                label="Download All Outputs",
                data=combined_export,
                file_name="listing_marketing_package.txt",
                mime="text/plain",
                key="download_all_structured"
            )

        except Exception as e:
            st.error(f"Error: {e}")

elif page_mode == "Paste Existing Listing / Notes":
    with st.form("rewrite_form"):
        city = st.selectbox("City / Market Context", ["Riverton", "Lander", "Other"])
        tone = st.selectbox("Tone", ["professional", "warm", "upscale", "concise"])
        length = st.selectbox("Length", ["short", "medium", "long"])
        raw_text = st.text_area(
            "Paste Zillow text, MLS remarks, seller notes, or rough property details",
            height=260,
            placeholder="Paste the existing listing text or rough notes here..."
        )

        st.subheader("Choose what to generate")
        gen_col1, gen_col2, gen_col3 = st.columns(3)
        with gen_col1:
            generate_listing_rw = st.form_submit_button("Generate Listing Copy")
        with gen_col2:
            generate_marketing_rw = st.form_submit_button("Generate Marketing Package")
        with gen_col3:
            generate_both_rw = st.form_submit_button("Generate Both")

    if generate_listing_rw or generate_marketing_rw or generate_both_rw:
        try:
            if not api_key:
                st.error("Missing OPENAI_API_KEY in environment or deployment secrets.")
                st.stop()

            if not raw_text.strip():
                st.error("Paste some listing text or property notes first.")
                st.stop()

            all_examples = load_examples()
            selected_examples = select_examples(all_examples, city=city, max_examples=3)

            listing_output = None
            marketing_output = None

            if generate_listing_rw or generate_both_rw:
                with st.spinner("Generating listing copy..."):
                    listing_prompt = build_rewrite_listing_prompt(raw_text, city, tone, length, selected_examples)
                    listing_output = generate_json(listing_prompt)

            if generate_marketing_rw or generate_both_rw:
                with st.spinner("Generating marketing package..."):
                    marketing_prompt = build_rewrite_marketing_prompt(raw_text, city, tone, length, selected_examples)
                    marketing_output = generate_json(marketing_prompt)

            st.success("Done.")

            if listing_output:
                st.header("Listing Copy")
                render_output_block("Headline", listing_output["headline"], "headline_rw", "headline.txt", height=80)
                render_output_block("MLS Description", listing_output["mls_description"], "mls_rw", "mls_description.txt", height=220)
                render_output_block("Portal Description", listing_output["portal_description"], "portal_rw", "portal_description.txt", height=220)
                render_output_block("Social Caption", listing_output["social_caption"], "social_rw", "social_caption.txt", height=140)

            if marketing_output:
                st.header("Marketing Package")
                render_output_block("Facebook Post", marketing_output["facebook_post"], "facebook_rw", "facebook_post.txt", height=180)
                render_output_block("Instagram Caption", marketing_output["instagram_caption"], "instagram_rw", "instagram_caption.txt", height=160)
                render_output_block("Buyer Email", marketing_output["buyer_email"], "buyer_email_rw", "buyer_email.txt", height=220)
                render_output_block("SMS Text", marketing_output["sms_text"], "sms_rw", "sms_text.txt", height=120)
                render_highlights_block(marketing_output["property_highlights"], "highlights_rw")
                render_output_block("Open House Post", marketing_output["open_house_post"], "open_house_rw", "open_house_post.txt", height=180)

            combined_export = build_combined_export(listing_output, marketing_output, None)
            st.header("Export All")
            st.download_button(
                label="Download All Outputs",
                data=combined_export,
                file_name="listing_marketing_package.txt",
                mime="text/plain",
                key="download_all_rewrite"
            )

        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.subheader("Weekly Marketing Content")

    with st.form("weekly_content_form"):
        col1, col2 = st.columns(2)

        with col1:
            content_type = st.selectbox(
                "Content Type",
                [
                    "Market Update",
                    "Buyer Tip",
                    "Seller Tip",
                    "Homeownership Tip",
                    "Community Highlight",
                    "FAQ / Myth",
                    "Surprise Me",
                ]
            )
            area_focus = st.selectbox(
                "Area Focus",
                ["Riverton", "Lander", "Fremont County", "General"]
            )

        with col2:
            tone = st.selectbox(
                "Tone",
                ["Professional", "Warm", "Informative", "Engaging"]
            )
            graphic_size = st.selectbox(
                "Graphic Size",
                ["1024x1024", "1536x1024", "1024x1536"]
            )

        extra_notes = st.text_area(
            "Extra Notes / Guidance (optional)",
            height=160,
            placeholder="Example: focus on first-time buyers, spring market activity, or local lifestyle in Lander."
        )

        colA, colB = st.columns(2)
        with colA:
            generate_weekly = st.form_submit_button("Generate Weekly Content")
        with colB:
            generate_weekly_graphic = st.form_submit_button("Generate Weekly Content + Graphic")

    if generate_weekly or generate_weekly_graphic:
        try:
            if not api_key:
                st.error("Missing OPENAI_API_KEY in environment or deployment secrets.")
                st.stop()

            include_graphic = bool(generate_weekly_graphic)

            with st.spinner("Generating weekly marketing content..."):
                weekly_prompt = build_weekly_content_prompt(
                    content_type=content_type,
                    area_focus=area_focus,
                    tone=tone,
                    extra_notes=extra_notes,
                    include_graphic=include_graphic,
                )
                weekly_output = generate_json(weekly_prompt)

            st.success("Done.")

            st.header("Weekly Content")
            render_output_block("Topic", weekly_output["topic"], "weekly_topic", "weekly_topic.txt", height=80)
            render_output_block("Facebook Post", weekly_output["facebook_post"], "weekly_fb", "weekly_facebook_post.txt", height=180)
            render_output_block("Instagram Caption", weekly_output["instagram_caption"], "weekly_ig", "weekly_instagram_caption.txt", height=160)
            render_output_block("Short Video Script", weekly_output["short_video_script"], "weekly_video", "weekly_video_script.txt", height=220)

            image_bytes = None
            if include_graphic and clean_field(weekly_output.get("graphic_prompt", "")):
                with st.spinner("Generating graphic..."):
                    image_bytes = generate_image_bytes(
                        prompt=weekly_output["graphic_prompt"],
                        size=graphic_size
                    )

            if image_bytes:
                st.header("Generated Graphic")
                if clean_field(weekly_output.get("graphic_headline", "")):
                    st.markdown(f"**Graphic Headline:** {weekly_output['graphic_headline']}")
                st.image(image_bytes, use_container_width=True)
                st.download_button(
                    label="Download Graphic",
                    data=image_bytes,
                    file_name="weekly_marketing_graphic.png",
                    mime="image/png",
                    key="download_weekly_graphic"
                )

            combined_export = build_combined_export(None, None, weekly_output)
            st.header("Export All")
            st.download_button(
                label="Download Weekly Content",
                data=combined_export,
                file_name="weekly_marketing_content.txt",
                mime="text/plain",
                key="download_all_weekly"
            )

        except Exception as e:
            st.error(f"Error: {e}")