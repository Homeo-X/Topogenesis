// Forge System OS — LLM proxy Edge Function (Deno).
//
// Holds model API keys server-side (Supabase secrets), so the app binary never
// carries them. Verifies the caller's Supabase JWT (verify_jwt is on by
// default for deployed functions). Routes by `task` and tries a provider chain:
//   Groq (primary) -> Gemini (fallback) -> 502.
// On 502 the CLIENT's existing silent fallback to the deterministic engine
// engages, so a total provider outage degrades gracefully, never breaks.
//
// Per-provider adapters normalize each model's response into the exact JSON
// shape the client validators (validatePatch / mergeArc /
// validateOutcomeSuggestion) already expect, so no client validator changes.
//
// Secrets (set via `supabase secrets set`):
//   GROQ_API_KEY    (optional) — if absent, Groq is skipped
//   GEMINI_API_KEY  (optional) — if absent, Gemini is skipped
// At least one must be present for refinement; otherwise the function 502s and
// the client falls back to deterministic generation.

// deno-lint-ignore-file no-explicit-any

const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";
const GROQ_MODEL = "llama-3.3-70b-versatile"; // single place to update if deprecated
const GEMINI_URL =
  "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"; // OpenAI-compat endpoint
const GEMINI_MODEL = "gemini-2.0-flash";

interface Provider {
  name: string;
  url: string;
  model: string;
  key: string | undefined;
}

function providers(): Provider[] {
  return [
    { name: "groq", url: GROQ_URL, model: GROQ_MODEL, key: Deno.env.get("GROQ_API_KEY") },
    { name: "gemini", url: GEMINI_URL, model: GEMINI_MODEL, key: Deno.env.get("GEMINI_API_KEY") }
  ];
}

const SYSTEM_PROMPT =
  "You are Forge System OS, a consent-based real-life quest system. Return STRICT JSON only, " +
  "no prose, no markdown fences. Never shame the user. Never give medical, legal, or financial " +
  "directives. Keep any ids and structural fields exactly as provided in the input.";

// Build the user-facing instruction per task from the payload the client sent.
function instructionFor(task: string, payload: any): string {
  if (task === "quest") {
    return (
      "Refine ONLY the display fields of this quest. Keep risk, time, domain, mode, and rewards " +
      "unchanged. Return JSON: {title, objective, activityPlan:{intent, stakes, steps:[{id,label,minutes,instruction,output}], " +
      "successCriteria:[], fallback, antiAvoidanceRule}, proofRequired, systemMessage}. " +
      "Stakes must frame what the quest builds or protects, never loss or punishment.\n\nINPUT:\n" +
      JSON.stringify(payload)
    );
  }
  if (task === "arc") {
    return (
      "Refine this goal arc. Keep the same milestone ids, count, and order. Return JSON: " +
      "{title, currentPhase, bottleneck, weeklyFocus, milestones:[{id,label,objective}], nextActions:[]}.\n\nINPUT:\n" +
      JSON.stringify(payload)
    );
  }
  // outcome
  return (
    "Read the user's note about a completed quest. Classify what actually happened and name the " +
    "main blocker gently. Return JSON: {suggestedOutcome, blocker, reasoning}. suggestedOutcome must " +
    "be one of the allowedOutcomes in the input.\n\nINPUT:\n" +
    JSON.stringify(payload)
  );
}

// Strip accidental markdown fences and parse JSON.
function extractJson(text: string): any {
  const cleaned = text.replace(/```json/gi, "").replace(/```/g, "").trim();
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  if (start === -1 || end === -1) throw new Error("No JSON object in model output");
  return JSON.parse(cleaned.slice(start, end + 1));
}

// Both providers are OpenAI chat-completions compatible, so one caller works;
// the adapter is the response-shape normalization (both return choices[0].message.content).
async function callProvider(p: Provider, task: string, payload: any): Promise<any> {
  const response = await fetch(p.url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${p.key}`
    },
    body: JSON.stringify({
      model: p.model,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: instructionFor(task, payload) }
      ],
      temperature: 0.7,
      // Groq honors response_format; Gemini's compat endpoint tolerates it.
      response_format: { type: "json_object" }
    })
  });
  if (!response.ok) {
    throw new Error(`${p.name} HTTP ${response.status}`);
  }
  const data = await response.json();
  const content = data?.choices?.[0]?.message?.content ?? "";
  return extractJson(String(content));
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "POST only" }), {
      status: 405,
      headers: { "Content-Type": "application/json" }
    });
  }

  let body: any;
  try {
    body = await req.json();
  } catch {
    return new Response(JSON.stringify({ error: "Invalid JSON body" }), {
      status: 400,
      headers: { "Content-Type": "application/json" }
    });
  }

  const task = body?.task;
  const payload = body?.payload;
  if (task !== "quest" && task !== "arc" && task !== "outcome") {
    return new Response(JSON.stringify({ error: "Unknown task" }), {
      status: 400,
      headers: { "Content-Type": "application/json" }
    });
  }

  // Try the provider chain in order; first success wins.
  const chain = providers().filter((p) => !!p.key);
  if (chain.length === 0) {
    return new Response(JSON.stringify({ error: "No provider keys configured" }), {
      status: 502,
      headers: { "Content-Type": "application/json" }
    });
  }

  const errors: string[] = [];
  for (const p of chain) {
    try {
      const result = await callProvider(p, task, payload);
      return new Response(JSON.stringify(result), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      });
    } catch (e) {
      errors.push(e instanceof Error ? e.message : String(e));
      // try next provider
    }
  }

  // All providers failed -> 502 -> client falls back to deterministic engine.
  return new Response(JSON.stringify({ error: "All providers failed", detail: errors }), {
    status: 502,
    headers: { "Content-Type": "application/json" }
  });
});
