import { useEffect, useMemo, useState } from "react";
import {
  ApiError,
  fetchProposedRecipe,
  recipeExportUrl,
  reviseRecipe,
  saveProposedRecipe,
  setProposedRecipeStatus,
  type ProposedIngredient,
  type ProposedRecipe,
  type ProposedIngredientEdit,
  type ProposedMethodEdit,
  type RevisionSuggestion,
} from "./api";

interface DraftRow {
  name: string;
  quantity: string;
  unit: string;
}

interface Strings {
  ingredients: string;
  method: string;
  notes: string;
  ingredient: string;
  quantity: string;
  unit: string;
  source: string;
  notDetermined: string;
  save: string;
  saving: string;
  approve: string;
  exportPdf: string;
  exportDocx: string;
  draft: string;
  approved: string;
  improve: string;
  tellAi: string;
  tellAiPlaceholder: string;
  generate: string;
  generating: string;
  reviseNeedsInput: string;
  reviseNeedsValues: string;
  reviseFailed: string;
  allowIngredients: string;
  aiSuggestion: string;
  basedOn: string;
  languageTag: string;
}

const STRINGS: Record<string, Strings> = {
  en: {
    ingredients: "Ingredients",
    method: "Method",
    notes: "Production notes",
    ingredient: "Ingredient",
    quantity: "Quantity",
    unit: "Unit",
    source: "Source",
    notDetermined: "Not determined",
    save: "Save changes",
    saving: "Saving…",
    approve: "Approve recipe",
    exportPdf: "Export PDF",
    exportDocx: "Export DOCX",
    draft: "Draft",
    approved: "Approved",
    improve: "Improve Your Recipe",
    tellAi: "Tell AI What to Improve",
    tellAiPlaceholder:
      "e.g. Add baking instructions and make the method easier " +
      "for my kitchen team. Do not change ingredient quantities.",
    generate: "Generate New Version",
    generating: "Generating…",
    reviseNeedsInput:
      "Select at least one improvement with a value, or " +
      "describe what to improve.",
    reviseNeedsValues:
      "Enter a value for each selected improvement, or " +
      "describe what to improve.",
    reviseFailed:
      "Could not generate a new version. Please try again.",
    allowIngredients:
      "Allow AI to change ingredient names and quantities",
    aiSuggestion: "AI suggestion",
    basedOn: "Based on",
    languageTag: "English",
  },
  es: {
    ingredients: "Ingredientes",
    method: "Preparación",
    notes: "Notas de producción",
    ingredient: "Ingrediente",
    quantity: "Cantidad",
    unit: "Unidad",
    source: "Origen",
    notDetermined: "No determinado",
    save: "Guardar cambios",
    saving: "Guardando…",
    approve: "Aprobar receta",
    exportPdf: "Exportar PDF",
    exportDocx: "Exportar DOCX",
    draft: "Borrador",
    approved: "Aprobada",
    improve: "Mejora tu receta",
    tellAi: "Indica qué quieres mejorar",
    tellAiPlaceholder:
      "p. ej. Agrega instrucciones de horneado y simplifica el " +
      "método para mi equipo. No cambies las cantidades.",
    generate: "Generar nueva versión",
    generating: "Generando…",
    reviseNeedsInput:
      "Selecciona al menos una mejora con su valor o " +
      "describe qué quieres mejorar.",
    reviseNeedsValues:
      "Escribe el valor de cada mejora seleccionada o " +
      "describe qué quieres mejorar.",
    reviseFailed:
      "No se pudo generar la nueva versión. Inténtalo de nuevo.",
    allowIngredients:
      "Permitir que la IA cambie ingredientes y cantidades",
    aiSuggestion: "Sugerencia de IA",
    basedOn: "Basada en",
    languageTag: "Español",
  },
};

const VISUAL_HINTS = [
  "Displayed",
  "display",
  "observed",
  "Observed",
  "Counted",
  "counted",
];
const AUDIO_HINTS = [
  "spoken",
  "Spoken",
  "mentioned",
  "Mentioned",
  "narration",
  "speaker",
  "audio",
];

function sourceLabel(
  row: ProposedIngredient,
  language: string,
): string {
  const es = language === "es";
  if (row.chef_edited) return es ? "Editado por el chef" : "Edited by chef";
  if (row.quantity_status === "ai_estimated") {
    return es ? "Sugerido por IA" : "Suggested by AI";
  }
  if (row.quantity_status === "seen_video") {
    return es ? "Visto en el video" : "Seen in video";
  }
  if (row.quantity_status === "mentioned_audio") {
    return es ? "Mencionado en el audio" : "Mentioned in audio";
  }
  if (
    row.quantity_status === "chef_supplied" ||
    row.quantity_status === "observed_count"
  ) {
    const basis = row.quantity_basis ?? "";
    const kinds = (row.evidence ?? [])
      .map((entry) => entry.kind ?? "")
      .join(" ");
    const visual =
      VISUAL_HINTS.some((hint) => basis.includes(hint)) ||
      kinds.includes("scale_display");
    const audio = AUDIO_HINTS.some((hint) => basis.includes(hint));
    if (visual && audio) {
      return es
        ? "Visto en el video + Mencionado en el audio"
        : "Seen in video + Mentioned in audio";
    }
    if (audio) return es ? "Mencionado en el audio" : "Mentioned in audio";
    if (visual) return es ? "Visto en el video" : "Seen in video";
    return es ? "Desconocido" : "Unknown";
  }
  return es ? "Desconocido" : "Unknown";
}

function adjustmentDetail(
  row: ProposedIngredient,
  language: string,
): string | null {
  const adjustment = row.draft_adjustment;
  if (!adjustment) return null;
  const shown = adjustment.displayed_value;
  const draft = adjustment.draft_value;
  if (shown === null || draft === null) return null;
  const unit = row.unit ?? "";
  if (language === "es") {
    return (
      `La báscula mostró ${shown} ${unit}; se mencionaron ` +
      `${draft} ${unit} en el audio.`
    );
  }
  return (
    `Scale showed ${shown} ${unit}; ${draft} ${unit} was ` +
    "mentioned in the audio."
  );
}

function toDraft(recipe: ProposedRecipe): {
  rows: DraftRow[];
  steps: string[];
} {
  return {
    rows: recipe.ingredients.map((row) => ({
      name: row.name,
      quantity:
        row.quantity === null || row.quantity === undefined
          ? ""
          : String(row.quantity),
      unit: row.unit ?? "",
    })),
    steps: recipe.method.map((step) => step.instruction),
  };
}

export default function RecipeEditor(props: {
  recipeId: string;
  onNewVersion: (recipeId: string) => void;
}) {
  const { recipeId } = props;
  const [recipe, setRecipe] = useState<ProposedRecipe | null>(null);
  const [rows, setRows] = useState<DraftRow[]>([]);
  const [steps, setSteps] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [instructions, setInstructions] = useState("");
  const [allowIngredients, setAllowIngredients] = useState(false);
  const [revising, setRevising] = useState(false);
  const [selections, setSelections] = useState<
    Record<string, { selected: boolean; value: string }>
  >({});

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setRecipe(null);
    setError(null);
    setNotice(null);
    setInstructions("");
    setSelections({});
    fetchProposedRecipe(recipeId)
      .then((data) => {
        if (cancelled) return;
        setRecipe(data);
        const draft = toDraft(data);
        setRows(draft.rows);
        setSteps(draft.steps);
        setError(null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(
          err instanceof ApiError
            ? `Could not load the recipe (${err.message}).`
            : "Could not load the recipe.",
        );
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [recipeId]);

  const dirty = useMemo(() => {
    if (!recipe) return false;
    if (rows.length !== recipe.ingredients.length) return true;
    if (steps.length !== recipe.method.length) return true;
    for (let i = 0; i < rows.length; i += 1) {
      const saved = recipe.ingredients[i];
      const savedQty =
        saved.quantity === null || saved.quantity === undefined
          ? ""
          : String(saved.quantity);
      if (
        rows[i].name !== saved.name ||
        rows[i].quantity !== savedQty ||
        rows[i].unit !== (saved.unit ?? "")
      )
        return true;
    }
    for (let i = 0; i < steps.length; i += 1) {
      if (steps[i] !== recipe.method[i].instruction) return true;
    }
    return false;
  }, [recipe, rows, steps]);

  async function handleSave(): Promise<void> {
    if (!recipe || saving) return;
    setError(null);
    setNotice(null);
    const ingredients: ProposedIngredientEdit[] = [];
    for (let i = 0; i < rows.length; i += 1) {
      const saved = recipe.ingredients[i];
      const savedQty =
        saved.quantity === null || saved.quantity === undefined
          ? ""
          : String(saved.quantity);
      const edit: ProposedIngredientEdit = { index: i };
      let changed = false;
      if (rows[i].name !== saved.name) {
        if (!rows[i].name.trim()) {
          setError(`Ingredient ${i + 1}: name cannot be empty.`);
          return;
        }
        edit.name = rows[i].name.trim();
        changed = true;
      }
      if (rows[i].quantity !== savedQty) {
        const raw = rows[i].quantity.trim();
        if (raw === "") {
          edit.quantity = null;
        } else {
          const value = Number(raw);
          if (!Number.isFinite(value) || value < 0) {
            setError(
              `Ingredient ${i + 1}: quantity must be a number or empty.`,
            );
            return;
          }
          edit.quantity = value;
        }
        changed = true;
      }
      if (rows[i].unit !== (saved.unit ?? "")) {
        edit.unit = rows[i].unit.trim();
        changed = true;
      }
      if (changed) ingredients.push(edit);
    }
    const method: ProposedMethodEdit[] = [];
    for (let i = 0; i < steps.length; i += 1) {
      if (steps[i] !== recipe.method[i].instruction) {
        if (!steps[i].trim()) {
          setError(`Step ${i + 1}: instruction cannot be empty.`);
          return;
        }
        method.push({
          number: recipe.method[i].number,
          instruction: steps[i].trim(),
        });
      }
    }
    if (ingredients.length === 0 && method.length === 0) {
      setNotice("No changes to save.");
      return;
    }
    setSaving(true);
    try {
      const updated = await saveProposedRecipe(recipeId, {
        ingredients,
        method,
      });
      setRecipe(updated);
      const draft = toDraft(updated);
      setRows(draft.rows);
      setSteps(draft.steps);
      setNotice("Changes saved.");
    } catch (err: unknown) {
      setError(
        err instanceof ApiError
          ? `Could not save (${err.message}).`
          : "Could not save.",
      );
    } finally {
      setSaving(false);
    }
  }

  async function handleApprove(): Promise<void> {
    if (!recipe || saving) return;
    if (dirty) {
      setError("Save your changes before approving the recipe.");
      return;
    }
    setError(null);
    setNotice(null);
    setSaving(true);
    try {
      const updated = await setProposedRecipeStatus(recipeId, {
        status: "approved",
      });
      setRecipe(updated);
      const draft = toDraft(updated);
      setRows(draft.rows);
      setSteps(draft.steps);
      setNotice("Recipe approved.");
    } catch (err: unknown) {
      setError(
        err instanceof ApiError
          ? `Could not approve (${err.message}).`
          : "Could not approve.",
      );
    } finally {
      setSaving(false);
    }
  }

  async function handleRevise(): Promise<void> {
    if (!recipe || revising || saving) return;
    if (dirty) {
      setError("Save your changes before generating a new version.");
      return;
    }
    const strings = STRINGS[recipe.language === "es" ? "es" : "en"];
    const choices = recipe.improvement_suggestions ?? [];
    const hasInstructions = instructions.trim().length > 0;
    const valued = choices.filter(
      (item) =>
        selections[item.id]?.selected &&
        (selections[item.id]?.value ?? "").trim().length > 0,
    );
    const selectedEmpty = choices.filter(
      (item) =>
        selections[item.id]?.selected &&
        (selections[item.id]?.value ?? "").trim().length === 0,
    );
    if (!hasInstructions && valued.length === 0) {
      setError(
        selectedEmpty.length > 0
          ? strings.reviseNeedsValues
          : strings.reviseNeedsInput,
      );
      return;
    }
    setError(null);
    setNotice(null);
    setRevising(true);
    const suggestions: RevisionSuggestion[] = choices.map((item) => ({
      id: item.id,
      selected: selections[item.id]?.selected ?? false,
      value: selections[item.id]?.value?.trim()
        ? selections[item.id].value.trim()
        : null,
    }));
    try {
      const result = await reviseRecipe(recipeId, {
        instructions: instructions.trim(),
        suggestions,
        allow_ingredient_changes: allowIngredients,
        confirmed: true,
      });
      props.onNewVersion(result.recipe_id);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        console.error("Generate New Version failed:", err.message);
      }
      setError(strings.reviseFailed);
    } finally {
      setRevising(false);
    }
  }

  if (loading) {
    return (
      <section className="card rc-wrap" aria-label="Recipe">
        <h2>Recipe</h2>
        <p className="muted">Loading the proposed recipe…</p>
      </section>
    );
  }

  if (!recipe) {
    return (
      <section className="card rc-wrap" aria-label="Recipe">
        <h2>Recipe</h2>
        {error && (
          <p className="rc-error" role="alert">
            {error}
          </p>
        )}
      </section>
    );
  }

  const language = recipe.language === "es" ? "es" : "en";
  const t = STRINGS[language];
  const statusLabel =
    recipe.recipe_status === "approved" ? t.approved : t.draft;
  const suggestions = recipe.improvement_suggestions ?? [];

  return (
    <div className="rc-doc">
      <header className="rc-head">
        <div>
          <p className="rc-kicker">
            AI-generated draft · {t.languageTag}
            {recipe.parent_name &&
              ` · ${t.basedOn} ${recipe.parent_name}`}
          </p>
          <h2 className="rc-title">
            {recipe.display_name || recipe.recipe_name || recipe.title}
          </h2>
          <p className="muted small">
            {recipe.ingredients.length} {t.ingredients.toLowerCase()} ·{" "}
            {recipe.method.length} {t.method.toLowerCase()}
            {recipe.last_saved_at &&
              ` · Last saved ${recipe.last_saved_at}`}
          </p>
        </div>
        <div className="rc-actions">
          <span
            className={`rc-status rc-status-${recipe.recipe_status}`}
            role="status"
          >
            {statusLabel}
          </span>
          <button
            type="button"
            className="rc-btn rc-btn-primary"
            disabled={saving || revising || !dirty}
            onClick={() => void handleSave()}
          >
            {saving ? t.saving : t.save}
          </button>
          <button
            type="button"
            className="rc-btn"
            disabled={saving || revising}
            onClick={() => void handleRevise()}
          >
            {revising ? t.generating : t.generate}
          </button>
          <button
            type="button"
            className="rc-btn"
            disabled={
              saving || revising || recipe.recipe_status === "approved"
            }
            onClick={() => void handleApprove()}
          >
            {t.approve}
          </button>
          <a
            className="rc-btn rc-btn-link"
            href={recipeExportUrl(recipeId, "pdf")}
            download
          >
            {t.exportPdf}
          </a>
          <a
            className="rc-btn rc-btn-link"
            href={recipeExportUrl(recipeId, "docx")}
            download
          >
            {t.exportDocx}
          </a>
        </div>
      </header>

      {dirty && (
        <p className="muted small" role="status">
          You have unsaved changes.
        </p>
      )}
      {error && (
        <p className="rc-error" role="alert">
          {error}
        </p>
      )}
      {notice && (
        <p className="rc-notice" role="status">
          {notice}
        </p>
      )}

      <h3 className="rc-section">{t.ingredients}</h3>
      <div className="rc-table-wrap">
        <table className="rc-table">
          <thead>
            <tr>
              <th scope="col">{t.ingredient}</th>
              <th scope="col">{t.quantity}</th>
              <th scope="col">{t.unit}</th>
              <th scope="col">{t.source}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => {
              const saved = recipe.ingredients[index];
              const detail = adjustmentDetail(saved, language);
              return (
                <tr key={`ing-${index}`}>
                  <td data-label={t.ingredient}>
                    <textarea
                      rows={1}
                      aria-label={`${t.ingredient} ${index + 1} name`}
                      value={row.name}
                      maxLength={120}
                      onChange={(event) => {
                        const next = rows.slice();
                        next[index] = {
                          ...next[index],
                          name: event.target.value,
                        };
                        setRows(next);
                      }}
                    />
                  </td>
                  <td data-label={t.quantity}>
                    <input
                      type="text"
                      inputMode="decimal"
                      aria-label={`${t.ingredient} ${index + 1} ${t.quantity.toLowerCase()}`}
                      placeholder={t.notDetermined}
                      value={row.quantity}
                      onChange={(event) => {
                        const next = rows.slice();
                        next[index] = {
                          ...next[index],
                          quantity: event.target.value,
                        };
                        setRows(next);
                      }}
                    />
                  </td>
                  <td data-label={t.unit}>
                    <input
                      type="text"
                      aria-label={`${t.ingredient} ${index + 1} ${t.unit.toLowerCase()}`}
                      value={row.unit}
                      maxLength={20}
                      onChange={(event) => {
                        const next = rows.slice();
                        next[index] = {
                          ...next[index],
                          unit: event.target.value,
                        };
                        setRows(next);
                      }}
                    />
                  </td>
                  <td data-label={t.source}>
                    <span
                      className={`rc-pill rc-pill-${saved.quantity_status}`}
                      title={saved.quantity_basis ?? undefined}
                    >
                      {sourceLabel(saved, language)}
                    </span>
                    {detail && (
                      <span className="muted small rc-detail">
                        {" "}
                        {detail}
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <h3 className="rc-section">{t.method}</h3>
      <ol className="rc-steps">
        {steps.map((text, index) => {
          const saved = recipe.method[index];
          return (
            <li key={`step-${saved.number}`} className="rc-step">
              <div className="rc-step-head">
                <span className="rc-step-no">{saved.number}</span>
                {saved.chef_edited && (
                  <span className="rc-pill rc-pill-chef">
                    {language === "es"
                      ? "Editado por el chef"
                      : "Edited by chef"}
                  </span>
                )}
              </div>
              <textarea
                aria-label={`Step ${saved.number} instruction`}
                value={text}
                rows={3}
                maxLength={2000}
                onChange={(event) => {
                  const next = steps.slice();
                  next[index] = event.target.value;
                  setSteps(next);
                }}
              />
              {saved.ai_guidance && (
                <p className="muted small rc-guidance">
                  {t.aiSuggestion}: {saved.ai_guidance.text}
                </p>
              )}
            </li>
          );
        })}
      </ol>

      <h3 className="rc-section">{t.notes}</h3>
      <ul className="rc-notes">
        {recipe.limitations.map((item) => (
          <li key={item.slice(0, 48)}>{item}</li>
        ))}
      </ul>

      {suggestions.length > 0 && (
        <>
          <h3 className="rc-section">{t.improve}</h3>
          <ul className="rc-suggest">
            {suggestions.map((item) => {
              const label =
                item.label ??
                (language === "es" ? item.label_es : item.label_en) ??
                "";
              const hint =
                item.hint ??
                (language === "es" ? item.hint_es : item.hint_en) ??
                "";
              const placeholder =
                item.placeholder ??
                (language === "es"
                  ? item.placeholder_es
                  : item.placeholder_en) ??
                "";
              const state = selections[item.id] ?? {
                selected: false,
                value: "",
              };
              return (
                <li key={item.id} className="rc-suggest-item">
                  <label className="rc-suggest-check">
                    <input
                      type="checkbox"
                      checked={state.selected}
                      onChange={(event) => {
                        setSelections({
                          ...selections,
                          [item.id]: {
                            ...state,
                            selected: event.target.checked,
                          },
                        });
                      }}
                    />
                    <span>
                      <strong>{label}</strong>
                      <span className="muted small"> {hint}</span>
                    </span>
                  </label>
                  {state.selected && (
                    <input
                      type="text"
                      className="rc-suggest-input"
                      aria-label={`${label} details`}
                      placeholder={placeholder}
                      value={state.value}
                      maxLength={500}
                      onChange={(event) => {
                        setSelections({
                          ...selections,
                          [item.id]: {
                            ...state,
                            value: event.target.value,
                          },
                        });
                      }}
                    />
                  )}
                </li>
              );
            })}
          </ul>
        </>
      )}

      <h3 className="rc-section">{t.tellAi}</h3>
      <textarea
        className="rc-revise-box"
        aria-label={t.tellAi}
        value={instructions}
        rows={3}
        maxLength={2000}
        placeholder={t.tellAiPlaceholder}
        onChange={(event) => setInstructions(event.target.value)}
      />
      <label className="rc-check-row">
        <input
          type="checkbox"
          checked={allowIngredients}
          onChange={(event) =>
            setAllowIngredients(event.target.checked)
          }
        />
        <span className="muted small">{t.allowIngredients}</span>
      </label>
      <p className="muted small">
        Generating a version uses one bounded AI call and never
        changes this version.
      </p>
    </div>
  );
}
