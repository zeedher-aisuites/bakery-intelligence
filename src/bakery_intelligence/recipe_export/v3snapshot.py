"""Proposed v3 recipe: chef-readable text, same evidence.

Deterministic transform of the v2 snapshot. Method
instructions and production notes are rewritten in plain
culinary language (English and Spanish); the original
technical wording, clip references, intervals, caveats, and
evidence links stay in each record. No inference, no edited
quantities, no new claims. v1 and v2 artifacts untouched.
"""

from __future__ import annotations

import copy
from typing import Any

V3_SNAPSHOT_ID = "bakery_demo_v1_proposed_v3"
V3_VERSION = "recipe-snapshot/3"

# Chef-readable instructions keyed by step number. Each entry is
# (english, spanish). Values stay within the recorded evidence;
# genuine unknowns become brief actionable guidance.
CLEAN_INSTRUCTIONS: dict[int, tuple[str, str]] = {
    1: ("Tare the scale and add a small flour portion to the bowl "
        "for the tangzhong (about 40 g was mentioned).",
        "Tara la báscula y agrega una porción pequeña de harina "
        "al tazón para el tangzhong (se mencionaron unos 40 g)."),
    2: ("Tare the scale and pour in 200 g of water, for a "
        "tangzhong total of about 240 g.",
        "Tara la báscula y vierte 200 g de agua, para un total "
        "de tangzhong de unos 240 g."),
    3: ("Mix the flour and water together until smooth.",
        "Mezcla la harina con el agua hasta lograr una mezcla "
        "homogénea."),
    4: ("Heat in the microwave in 30-40 second intervals, "
        "stirring thoroughly between intervals until the mixture "
        "thickens to a smooth paste.",
        "Calienta en el microondas en intervalos de 30-40 "
        "segundos, revolviendo bien entre cada intervalo hasta "
        "obtener una pasta espesa y suave."),
    5: ("Cover the paste with plastic wrap and chill until cold.",
        "Cubre la pasta con plástico adherente y refrigera hasta "
        "que esté fría."),
    6: ("Tare the scale with the steel mixing bowl and pour in "
        "milk from the red mug, about 240 g as mentioned during "
        "preparation.",
        "Tara la báscula con el tazón de acero y vierte leche "
        "del recipiente rojo, unos 240 g según lo mencionado "
        "durante la preparación."),
    7: ("Tare the scale and add 14 g of dry yeast from the small "
        "white dish.",
        "Tara la báscula y agrega 14 g de levadura seca del "
        "plato blanco pequeño."),
    8: ("Rest the mixture so the yeast can activate, until "
        "bubbles appear.",
        "Deja reposar la mezcla para que la levadura se active, "
        "hasta que aparezcan burbujas."),
    9: ("Place the bubbly fermented mass in the mixing bowl.",
        "Coloca la masa fermentada y burbujeante en el tazón de "
        "mezclado."),
    10: ("Weigh 100 g of sugar and add it.",
         "Pesa 100 g de azúcar y agrégala."),
    11: ("Stir in the chilled tangzhong paste.",
         "Incorpora la pasta de tangzhong fría."),
    12: ("Crack 2 eggs, add a cap of vanilla essence, whisk with "
         "a little liquid, and pour the egg mixture into the bowl.",
         "Casca 2 huevos, agrega una tapa de esencia de vainilla, "
         "bate con un poco de líquido y vierte la mezcla en el "
         "tazón."),
    13: ("Weigh 24 g of milk powder and add it.",
         "Pesa 24 g de leche en polvo y agrégala."),
    14: ("Weigh 700 g of bread flour and add it.",
         "Pesa 700 g de harina de pan y agrégala."),
    15: ("Weigh 3 g of salt and add it last.",
         "Pesa 3 g de sal y agrégala al final."),
    16: ("Mix with the stand mixer until the ingredients come "
         "together.",
         "Mezcla con la batidora de pedestal hasta integrar los "
         "ingredientes."),
    17: ("Weigh 113 g of butter and add it to the dough.",
         "Pesa 113 g de mantequilla y agrégala a la masa."),
    18: ("Machine-knead until the butter is fully incorporated.",
         "Amasa a máquina hasta incorporar por completo la "
         "mantequilla."),
    19: ("Check gluten development with the windowpane test.",
         "Verifica el desarrollo del gluten con la prueba de la "
         "membrana."),
    20: ("Finish kneading by hand until smooth.",
         "Termina de amasar a mano hasta lograr una masa suave."),
}

CHEF_NOTES: tuple[tuple[str, str], ...] = (
    ("No verified reference recipe exists yet; confirm the key "
     "quantities before scaling up to a full batch.",
     "Aún no existe una receta de referencia verificada; "
     "confirma las cantidades clave antes de una producción "
     "completa."),
    ("The two videos were recorded separately with an unknown "
     "gap; milk, yeast, and tangzhong carry over from the first "
     "video into the second.",
     "Los dos videos se grabaron por separado con un intervalo "
     "desconocido; la leche, la levadura y el tangzhong pasan "
     "del primer video al segundo."),
    ("The flour portion, milk, vanilla, tangzhong, activated "
     "yeast, and egg mixture follow what was mentioned during "
     "preparation; not all were weighed on camera.",
     "La porción de harina, la leche, la vainilla, el tangzhong, "
     "la levadura activada y la mezcla de huevo siguen lo "
     "mencionado durante la preparación; no todo se pesó en "
     "cámara."),
    ("Microwave settings, chilling time and temperature, mixer "
     "speed, resting time, and shaping were not fully shown; "
     "use the suggestions below to complete them.",
     "Los ajustes del microondas, el tiempo y la temperatura de "
     "enfriado, la velocidad de la batidora, el reposo y el "
     "formado no se mostraron por completo; usa las "
     "sugerencias para completarlos."),
    ("Butter type and flour type were not confirmed by their "
     "labels; confirm before preparing a new batch.",
     "El tipo de mantequilla y de harina no se confirmó con "
     "sus etiquetas; verifícalos antes de un nuevo lote."),
    ("A few scale displays were hard to read; values marked as "
     "AI suggestions still need your confirmation.",
     "Algunas pantallas de la báscula fueron difíciles de leer; "
     "los valores marcados como sugerencias de IA aún "
     "necesitan tu confirmación."),
    ("This draft was assembled by AI with an AI-assisted review "
     "pass; it still needs your approval before use.",
     "Este borrador fue elaborado por IA con una revisión "
     "asistida por IA; aún necesita tu aprobación antes de "
     "usarlo."),
)


def build_v3(v2: dict[str, Any], generated_at: str) -> dict[str, Any]:
    """Rewrite v2 prose for chefs; evidence fields untouched."""
    snapshot = copy.deepcopy(v2)
    snapshot["snapshot_version"] = V3_VERSION
    snapshot["snapshot_id"] = V3_SNAPSHOT_ID
    snapshot["generated_at"] = generated_at
    snapshot["recipe_status"] = "draft"
    snapshot["status_note"] = None
    snapshot["status_updated_at"] = None
    snapshot["last_saved_at"] = None
    snapshot["provenance"] = dict(v2.get("provenance") or {})
    snapshot["provenance"]["v3_parent"] = (
        "runs_api/recipes/bakery_demo_v1_proposed_v2/snapshot.json")
    snapshot["provenance"]["v3_note"] = (
        "Chef-readable rewrite of v2 prose (EN/ES). Original "
        "technical wording preserved per record; quantities, "
        "identities, and evidence links unchanged.")
    for item in snapshot.get("method") or []:
        number = item.get("number")
        pair = CLEAN_INSTRUCTIONS.get(number)
        if pair is None:
            raise ValueError(f"No clean text for step {number!r}.")
        item["technical_instruction"] = item.get("instruction")
        item["instruction_en"], item["instruction_es"] = pair
        item["instruction"] = pair[0]
    snapshot["technical_limitations"] = list(
        v2.get("limitations") or [])
    snapshot["notes_en"] = [pair[0] for pair in CHEF_NOTES]
    snapshot["notes_es"] = [pair[1] for pair in CHEF_NOTES]
    snapshot["limitations"] = list(snapshot["notes_en"])
    return snapshot
