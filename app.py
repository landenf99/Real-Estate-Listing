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
# Helpers
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


def build_rewrite_prompt(raw_text: str, city: str, tone: str, length: str, examples: list) -> str:
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


def generate_listing_copy(prompt: str) -> dict:
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
    data = {
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
    return data


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
        <div style="margin-top: 2px; margin-bottom: 2px;">
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


def build_combined_export(output: dict) -> str:
    return f"""HEADLINE
{output.get("headline", "")}

MLS DESCRIPTION
{output.get("mls_description", "")}

PORTAL DESCRIPTION
{output.get("portal_description", "")}

SOCIAL CAPTION
{output.get("social_caption", "")}
"""


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
st.set_page_config(page_title="Fremont County Listing Generator", page_icon="🏡")
st.title("Fremont County Residential Listing Generator")
st.caption("AI generator tuned to Riverton / Lander / Fremont County style.")

mode = st.radio(
    "Choose input mode",
    ["Structured Form", "Paste Existing Listing / Notes"],
    horizontal=True
)

if mode == "Structured Form":
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

        submitted = st.form_submit_button("Generate Listing Copy")

    if submitted:
        try:
            if not api_key:
                st.error("Missing OPENAI_API_KEY in .env file.")
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

            system_prompt = load_text_file(SYSTEM_PROMPT_FILE)
            user_prompt = build_user_prompt(property_data, selected_examples)

            final_prompt = f"""{system_prompt}

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

            with st.spinner("Generating listing copy..."):
                output = generate_listing_copy(final_prompt)

            st.success("Done.")

            render_output_block(
                "Headline",
                output["headline"],
                "headline",
                "headline.txt",
                height=80
            )
            render_output_block(
                "MLS Description",
                output["mls_description"],
                "mls",
                "mls_description.txt",
                height=220
            )
            render_output_block(
                "Portal Description",
                output["portal_description"],
                "portal",
                "portal_description.txt",
                height=220
            )
            render_output_block(
                "Social Caption",
                output["social_caption"],
                "social",
                "social_caption.txt",
                height=140
            )

            combined_export = build_combined_export(output)
            st.subheader("Export All")
            st.download_button(
                label="Download All Outputs",
                data=combined_export,
                file_name="listing_outputs.txt",
                mime="text/plain",
                key="download_all_structured"
            )

        except Exception as e:
            st.error(f"Error: {e}")

else:
    with st.form("rewrite_form"):
        city = st.selectbox("City / Market Context", ["Riverton", "Lander", "Other"])
        tone = st.selectbox("Tone", ["professional", "warm", "upscale", "concise"])
        length = st.selectbox("Length", ["short", "medium", "long"])
        raw_text = st.text_area(
            "Paste Zillow text, MLS remarks, seller notes, or rough property details",
            height=260,
            placeholder="Paste the existing listing text or rough notes here..."
        )

        submitted_rewrite = st.form_submit_button("Rewrite / Generate")

    if submitted_rewrite:
        try:
            if not api_key:
                st.error("Missing OPENAI_API_KEY in .env file.")
                st.stop()

            if not raw_text.strip():
                st.error("Paste some listing text or property notes first.")
                st.stop()

            all_examples = load_examples()
            selected_examples = select_examples(all_examples, city=city, max_examples=3)

            final_prompt = build_rewrite_prompt(
                raw_text=raw_text,
                city=city,
                tone=tone,
                length=length,
                examples=selected_examples,
            )

            with st.spinner("Rewriting listing..."):
                output = generate_listing_copy(final_prompt)

            st.success("Done.")

            render_output_block(
                "Headline",
                output["headline"],
                "headline_rw",
                "headline.txt",
                height=80
            )
            render_output_block(
                "MLS Description",
                output["mls_description"],
                "mls_rw",
                "mls_description.txt",
                height=220
            )
            render_output_block(
                "Portal Description",
                output["portal_description"],
                "portal_rw",
                "portal_description.txt",
                height=220
            )
            render_output_block(
                "Social Caption",
                output["social_caption"],
                "social_rw",
                "social_caption.txt",
                height=140
            )

            combined_export = build_combined_export(output)
            st.subheader("Export All")
            st.download_button(
                label="Download All Outputs",
                data=combined_export,
                file_name="listing_outputs.txt",
                mime="text/plain",
                key="download_all_rewrite"
            )

        except Exception as e:
            st.error(f"Error: {e}")