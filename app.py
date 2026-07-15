"""
Pantry Planner — turn what's in your kitchen into recipes, a weekly meal
plan, and a shopping list.

Nothing sensitive ever needs to leave your device here: the only data sent
to the Gemini API is a list of ingredients and food preferences (e.g.
"vegetarian", "no nuts") — no names, moods, health data, or personal
history involved.
"""

import io
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image

DB_PATH = Path(__file__).parent / "favorites.db"

DIETARY_OPTIONS = [
    "None", "Vegetarian", "Vegan", "Gluten-free", "Dairy-free",
    "Low-carb", "Nut-free", "Halal", "Kosher",
]

CUISINES = [
    "Any", "Italian", "Mexican", "Indian", "Chinese", "Thai",
    "Mediterranean", "American", "Japanese", "French",
]

# ----------------------------------------------------------------------
# Local storage for favorites (plain — nothing sensitive stored)
# ----------------------------------------------------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            title TEXT NOT NULL,
            recipe_json TEXT NOT NULL
        )"""
    )
    conn.commit()
    return conn


def save_favorite(conn, recipe: dict):
    conn.execute(
        "INSERT INTO favorites (ts, title, recipe_json) VALUES (?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), recipe["title"], json.dumps(recipe)),
    )
    conn.commit()


def fetch_favorites(conn):
    rows = conn.execute("SELECT id, title, recipe_json FROM favorites ORDER BY ts DESC").fetchall()
    return [(rid, title, json.loads(rj)) for rid, title, rj in rows]


def delete_favorite(conn, fav_id):
    conn.execute("DELETE FROM favorites WHERE id = ?", (fav_id,))
    conn.commit()


# ----------------------------------------------------------------------
# Gemini helpers
# ----------------------------------------------------------------------
RECIPE_SCHEMA_INSTRUCTIONS = """
You are a recipe assistant. Given a list of pantry ingredients and
preferences, respond with ONLY valid JSON (no markdown fences, no
commentary) matching this exact schema:

{
  "recipes": [
    {
      "title": string,
      "description": string (one sentence),
      "cuisine": string,
      "prep_time_minutes": integer,
      "servings": integer,
      "used_ingredients": [string, ...]   // pantry items this recipe uses
      "missing_ingredients": [string, ...] // needed items NOT in the pantry list
      "steps": [string, ...]
    }
  ]
}

Rules:
- Suggest recipes that primarily use the given pantry ingredients.
- missing_ingredients should be short shopping-list-style items (e.g. "soy sauce", not "2 tbsp soy sauce").
- Respect dietary restrictions strictly.
- Return exactly the number of recipes requested.
"""


def extract_json(text: str) -> dict:
    """Strip markdown fences if present and parse JSON safely."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


def generate_recipes(api_key, pantry, dietary, cuisine, count):
    client = genai.Client(api_key=api_key)
    prefs = []
    if dietary != "None":
        prefs.append(f"dietary restriction: {dietary}")
    if cuisine != "Any":
        prefs.append(f"preferred cuisine: {cuisine}")
    pref_text = "; ".join(prefs) if prefs else "no specific preferences"

    prompt = (
        f"Pantry ingredients: {', '.join(pantry)}.\n"
        f"Preferences: {pref_text}.\n"
        f"Generate exactly {count} recipe suggestions as JSON."
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=RECIPE_SCHEMA_INSTRUCTIONS,
            response_mime_type="application/json",
        ),
    )
    return extract_json(response.text)["recipes"]


def generate_meal_plan(api_key, pantry, dietary, days):
    client = genai.Client(api_key=api_key)
    prefs = f"dietary restriction: {dietary}" if dietary != "None" else "no specific preferences"
    prompt = (
        f"Pantry ingredients: {', '.join(pantry)}.\n"
        f"Preferences: {prefs}.\n"
        f"Generate a {days}-day dinner meal plan as JSON with exactly {days} recipes, "
        f"one per day, in the same schema."
    )
    system_instruction = (
        RECIPE_SCHEMA_INSTRUCTIONS
        + "\nGenerate one recipe per day, with good variety across the days."
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
        ),
    )
    return extract_json(response.text)["recipes"]


def detect_ingredients_from_image(api_key, image_bytes):
    """Send a photo to Gemini's vision model and get back a plain ingredient list."""
    client = genai.Client(api_key=api_key)
    img = Image.open(io.BytesIO(image_bytes))
    prompt = (
        "Look at this photo of a fridge, pantry, or countertop and identify every "
        "distinct food ingredient you can see. Respond with ONLY valid JSON in this "
        'exact schema: {"ingredients": [string, ...]}. Use simple everyday ingredient '
        'names (e.g. "carrots", "milk", "eggs"), not brand names, not quantities, no duplicates.'
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, img],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return extract_json(response.text)["ingredients"]


def merge_ingredients(existing_text: str, new_items: list) -> str:
    """Merge newly detected ingredients into the existing pantry text, deduping case-insensitively."""
    existing = [p.strip() for p in existing_text.split(",") if p.strip()]
    seen = {p.lower() for p in existing}
    for item in new_items:
        if item.strip().lower() not in seen:
            existing.append(item.strip())
            seen.add(item.strip().lower())
    return ", ".join(existing)


def friendly_error_message(e: Exception) -> str:
    """Turn raw API errors into something actionable, especially quota issues."""
    msg = str(e)
    if "429" in msg or "quota" in msg.lower() or "RESOURCE_EXHAUSTED" in msg:
        return (
            "**Rate limit / quota reached.** This is an account issue, not a bug in the "
            "app: your API key's project likely isn't linked to a Cloud Billing account, "
            "so Google routes it into a zero-quota bucket. Fix: link a billing account at "
            "[console.cloud.google.com/billing](https://console.cloud.google.com/billing), "
            "then generate a **new** API key (old keys can stay stuck on the old quota) at "
            "[aistudio.google.com/apikey](https://aistudio.google.com/apikey). "
            f"\n\nRaw error: `{msg[:300]}`"
        )
    return f"Error: {msg}"


# ----------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------
def render_recipe_card(recipe, conn, key_prefix):
    with st.container(border=True):
        st.markdown(f"### {recipe['title']}")
        st.caption(
            f"{recipe.get('cuisine', '')} · {recipe.get('prep_time_minutes', '?')} min · "
            f"serves {recipe.get('servings', '?')}"
        )
        st.write(recipe.get("description", ""))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ From your pantry**")
            for ing in recipe.get("used_ingredients", []):
                st.markdown(f"- {ing}")
        with col2:
            st.markdown("**🛒 You'll need to buy**")
            missing = recipe.get("missing_ingredients", [])
            if missing:
                for ing in missing:
                    st.markdown(f"- {ing}")
            else:
                st.markdown("_Nothing — you have it all!_")

        with st.expander("Steps"):
            for i, step in enumerate(recipe.get("steps", []), 1):
                st.markdown(f"{i}. {step}")

        if st.button("⭐ Save to favorites", key=f"{key_prefix}_save"):
            save_favorite(conn, recipe)
            st.toast(f"Saved '{recipe['title']}' to favorites!")


def render_photo_detector(api_key, pantry_key, widget_key):
    """Lets the user snap/upload a fridge or pantry photo; detected ingredients
    get merged into the pantry text area identified by pantry_key.

    The camera widget is only mounted once the user explicitly picks "Take a
    photo" — mounting st.camera_input eagerly (e.g. inside a tab that isn't
    currently visible) can leave it stuck initializing forever, since hidden
    tabs in Streamlit are CSS-hidden rather than unmounted.
    """
    with st.expander("📷 Or add ingredients from a photo"):
        mode = st.radio(
            "Add a photo",
            ["Upload a photo", "Take a photo"],
            key=f"{widget_key}_mode",
            horizontal=True,
            label_visibility="collapsed",
        )

        image_file = None
        if mode == "Upload a photo":
            image_file = st.file_uploader(
                "Upload a photo", type=["jpg", "jpeg", "png", "webp"],
                key=f"{widget_key}_upload", label_visibility="collapsed",
            )
        else:
            st.caption("Requires camera permission, and only works over HTTPS or localhost.")
            cam_enabled_key = f"{widget_key}_camera_enabled"
            if cam_enabled_key not in st.session_state:
                st.session_state[cam_enabled_key] = False

            if not st.session_state[cam_enabled_key]:
                if st.button("📷 Enable camera", key=f"{widget_key}_enable_cam"):
                    st.session_state[cam_enabled_key] = True
                    st.rerun()
            else:
                image_file = st.camera_input(
                    "Take a photo", key=f"{widget_key}_camera", label_visibility="collapsed",
                )
                if st.button("Turn off camera", key=f"{widget_key}_disable_cam"):
                    st.session_state[cam_enabled_key] = False
                    st.rerun()

        if image_file is not None:
            st.image(image_file, width=220)
            if st.button("Detect ingredients", key=f"{widget_key}_detect"):
                with st.spinner("Looking at your photo..."):
                    try:
                        found = detect_ingredients_from_image(api_key, image_file.getvalue())
                        if found:
                            st.session_state[pantry_key] = merge_ingredients(
                                st.session_state.get(pantry_key, ""), found
                            )
                            st.success(f"Added: {', '.join(found)}")
                        else:
                            st.warning("Couldn't identify any ingredients in that photo — try a clearer shot.")
                    except json.JSONDecodeError:
                        st.error("The model returned something that wasn't valid JSON. Try again.")
                    except Exception as e:
                        st.error(friendly_error_message(e))


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------
st.set_page_config(page_title="Pantry Planner", page_icon="🥗", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    h1, h2, h3, .app-title { font-family: 'Fraunces', serif; }
    .app-title { font-size: 2.1rem; font-weight: 600; color: #2E2A22; margin-bottom: 0; }
    .app-sub { color: #6B6455; font-size: 0.95rem; margin-top: 0.1rem; margin-bottom: 1.4rem; }
    </style>
    <div class="app-title">🥗 Pantry Planner</div>
    <div class="app-sub">Turn what's already in your kitchen into dinner.</div>
    """,
    unsafe_allow_html=True,
)

conn = get_conn()

api_key = st.secrets.get("GOOGLE_API_KEY", None) if hasattr(st, "secrets") else None

with st.sidebar:
    st.header("Settings")
    if not api_key:
        api_key = st.text_input("Google Gemini API key", type="password")
    else:
        st.success("API key loaded from secrets.toml")

    dietary = st.selectbox("Dietary preference", DIETARY_OPTIONS)
    cuisine = st.selectbox("Cuisine", CUISINES)

    st.divider()
    st.subheader("⭐ Favorites")
    favs = fetch_favorites(conn)
    if not favs:
        st.caption("No saved recipes yet.")
    for fid, title, recipe in favs:
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**{title}**")
        if c2.button("✕", key=f"del_{fid}"):
            delete_favorite(conn, fid)
            st.rerun()

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to get recipe suggestions.")
    st.stop()

tab1, tab2 = st.tabs(["🍳 Recipes from pantry", "📅 Weekly meal plan"])

# ----------------------------------------------------------------------
# Tab 1: quick recipes from a pantry list
# ----------------------------------------------------------------------
with tab1:
    if "pantry1" not in st.session_state:
        st.session_state["pantry1"] = ""

    render_photo_detector(api_key, pantry_key="pantry1", widget_key="tab1")

    pantry_text = st.text_area(
        "What's in your pantry / fridge?",
        placeholder="e.g. chicken thighs, rice, onion, garlic, soy sauce, bell peppers",
        height=100,
        key="pantry1",
    )
    count = st.slider("Number of recipe ideas", 1, 6, 3)

    if st.button("🔍 Find recipes", type="primary"):
        pantry = [p.strip() for p in pantry_text.split(",") if p.strip()]
        if not pantry:
            st.warning("Add at least one ingredient first.")
        else:
            with st.spinner("Thinking up some recipes..."):
                try:
                    recipes = generate_recipes(api_key, pantry, dietary, cuisine, count)
                    st.session_state["last_recipes"] = recipes
                except json.JSONDecodeError:
                    st.error("The model returned something that wasn't valid JSON. Try again.")
                except Exception as e:
                    st.error(friendly_error_message(e))

    if "last_recipes" in st.session_state:
        for i, recipe in enumerate(st.session_state["last_recipes"]):
            render_recipe_card(recipe, conn, key_prefix=f"recipe_{i}")

        all_missing = sorted({
            ing for r in st.session_state["last_recipes"] for ing in r.get("missing_ingredients", [])
        })
        if all_missing:
            st.divider()
            st.subheader("🛒 Combined shopping list")
            for ing in all_missing:
                st.markdown(f"- {ing}")
            st.download_button(
                "⬇️ Download shopping list (.txt)",
                "\n".join(all_missing),
                file_name="shopping_list.txt",
            )

# ----------------------------------------------------------------------
# Tab 2: weekly meal plan
# ----------------------------------------------------------------------
with tab2:
    if "pantry2" not in st.session_state:
        st.session_state["pantry2"] = ""

    render_photo_detector(api_key, pantry_key="pantry2", widget_key="tab2")

    pantry_text2 = st.text_area(
        "What's in your pantry / fridge?",
        placeholder="e.g. eggs, spinach, pasta, canned tomatoes, cheese, potatoes",
        height=100,
        key="pantry2",
    )
    days = st.slider("How many days?", 3, 7, 5)

    if st.button("📅 Plan my week", type="primary"):
        pantry2 = [p.strip() for p in pantry_text2.split(",") if p.strip()]
        if not pantry2:
            st.warning("Add at least one ingredient first.")
        else:
            with st.spinner("Planning your week..."):
                try:
                    plan = generate_meal_plan(api_key, pantry2, dietary, days)
                    st.session_state["last_plan"] = plan
                except json.JSONDecodeError:
                    st.error("The model returned something that wasn't valid JSON. Try again.")
                except Exception as e:
                    st.error(friendly_error_message(e))

    if "last_plan" in st.session_state:
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for i, recipe in enumerate(st.session_state["last_plan"]):
            st.markdown(f"#### {day_names[i] if i < len(day_names) else f'Day {i+1}'}")
            render_recipe_card(recipe, conn, key_prefix=f"plan_{i}")

        all_missing2 = sorted({
            ing for r in st.session_state["last_plan"] for ing in r.get("missing_ingredients", [])
        })
        if all_missing2:
            st.divider()
            st.subheader("🛒 Week's shopping list")
            for ing in all_missing2:
                st.markdown(f"- {ing}")
            st.download_button(
                "⬇️ Download shopping list (.txt)",
                "\n".join(all_missing2),
                file_name="weekly_shopping_list.txt",
            )
