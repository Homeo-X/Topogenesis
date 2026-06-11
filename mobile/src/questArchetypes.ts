import { QuestDomain } from "./types";

/**
 * Per-domain quest-archetype library.
 *
 * Design principles (every archetype must satisfy all five):
 *   1. BINARY completion — you instantly know if you did it.
 *   2. SMALL floor — the easiest variant is almost frictionless to start.
 *   3. TIME- or QUANTITY-BOXED — carries a concrete duration or count.
 *   4. BEHAVIORAL — rewards a controllable ACTION, never an outcome you can't will.
 *   5. ATOMIC — one core decision, not a disguised multi-step project.
 *
 * All content is original to Forge. The structure (verb-shape per domain, a
 * small/medium/large effort ladder, behavioral framing) is a general design
 * pattern, not copied content. Quests generated from these are deterministic
 * given the rotation index, so the engine stays testable.
 *
 * WELLBEING GUARDRAILS baked in:
 *   - No archetype names a weight/fat/aesthetic target (body = capability/behavior only).
 *   - No archetype uses restriction, shame, punishment, or coercion framing.
 *   - Social/body archetypes are consent-gated by the caller (preferences), not here.
 */

export type EffortTier = "small" | "medium" | "large";

export interface QuestArchetype {
  /** Stable id for testing / dedup. */
  id: string;
  /** The action template. {n} is replaced with the tier's quantity if present. */
  action: string;
  /** Characteristic verb shape of this archetype (for variety selection). */
  shape: VerbShape;
  /** Effort tier — drives time-box, XP scale, and difficulty band. */
  tier: EffortTier;
  /** Concrete minutes the execute step suggests (the time-box). */
  minutes: number;
  /** One-line "why this builds/protects" framing (never loss/threat/shame). */
  builds: string;
  /**
   * Per-domain level at which this archetype unlocks. Derived from the domain's
   * stat total (see domainLevel). Lower-level archetypes ALWAYS stay in the pool
   * once unlocked — they become repeatable habits; higher levels ADD options,
   * never replace. Defaults to 1 (available from the start).
   */
  unlockLevel?: number;
  /** Subcategory label, so growing depth reads as organized themes. */
  subcategory?: string;
}

export type VerbShape =
  | "do_a_rep" // perform one small repeatable action
  | "consume" // read/watch/listen/study one thing
  | "discover" // find/try something new
  | "reach_out" // one outward relational move
  | "regulate" // self-regulation / calming
  | "tidy_fix" // clear or repair one thing
  | "go_explore" // go somewhere / change route
  | "make_real" // produce one concrete artifact
  | "prepare" // set up / plan / organize a next step
  | "name_face"; // name or face an avoided thing (courage)

/**
 * The library. Each domain has multiple archetypes spanning effort tiers and
 * verb shapes, so generated quests feel varied and category-appropriate rather
 * than repetitive. Counts per domain are intentionally generous.
 */
export const QUEST_ARCHETYPES: Record<QuestDomain, QuestArchetype[]> = {
  craft: [
    // --- Foundations (L1) ---
    { id: "craft_smallest_piece", action: "Build the smallest piece that actually works.", shape: "make_real", tier: "small", minutes: 15, builds: "One real piece changes what you can do next.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "craft_rough_draft", action: "Make a rough version you're allowed to throw away.", shape: "make_real", tier: "small", minutes: 20, builds: "A throwaway draft takes the fear out of starting.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "craft_fix_one", action: "Fix one thing that's been broken or unfinished.", shape: "tidy_fix", tier: "small", minutes: 15, builds: "Clearing a small mess frees you up for the next thing.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "craft_one_commit", action: "Make one small change and save it properly.", shape: "make_real", tier: "small", minutes: 15, builds: "Small finished pieces are how big things get built.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "craft_clear_blocker", action: "Clear one thing that's stopping you from starting.", shape: "tidy_fix", tier: "small", minutes: 12, builds: "Getting out of your own way is half the work.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "craft_name_next", action: "Write down the very next step for what you're building.", shape: "prepare", tier: "small", minutes: 8, builds: "A clear next step turns a big project into a move.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "craft_ten_min", action: "Work on it for just 10 minutes, then stop if you want.", shape: "make_real", tier: "small", minutes: 10, builds: "Ten minutes is usually enough to get going.", unlockLevel: 1, subcategory: "Foundations" },
    // --- Practice (L2) ---
    { id: "craft_hardest_part", action: "Spend 25 minutes on the part you understand least.", shape: "make_real", tier: "medium", minutes: 25, builds: "The hard part is where the real progress hides.", unlockLevel: 2, subcategory: "Practice" },
    { id: "craft_prototype", action: "Make a quick, rough version just to see it work.", shape: "make_real", tier: "medium", minutes: 30, builds: "Something that runs teaches you more than a plan.", unlockLevel: 2, subcategory: "Practice" },
    { id: "craft_refactor_one", action: "Clean up one rough part without adding anything new.", shape: "tidy_fix", tier: "medium", minutes: 25, builds: "Tidying what's there makes the next part easier.", unlockLevel: 2, subcategory: "Practice" },
    { id: "craft_test_one", action: "Add one test that proves a piece really works.", shape: "make_real", tier: "medium", minutes: 20, builds: "A test you trust lets you move faster without worry.", unlockLevel: 2, subcategory: "Practice" },
    { id: "craft_document_one", action: "Write a few notes on how one part works.", shape: "make_real", tier: "small", minutes: 15, builds: "Writing it down shows you what you didn't fully get.", unlockLevel: 2, subcategory: "Practice" },
    { id: "craft_pair_review", action: "Show one piece to someone and ask them one question.", shape: "reach_out", tier: "medium", minutes: 20, builds: "Another set of eyes catches what yours skip over.", unlockLevel: 2, subcategory: "Practice" },
    { id: "craft_learn_tool", action: "Spend 25 minutes learning a tool that would speed things up.", shape: "consume", tier: "medium", minutes: 25, builds: "A better tool pays you back on every job after.", unlockLevel: 2, subcategory: "Practice" },
    // --- Craftsmanship (L5) ---
    { id: "craft_ship_one", action: "Finish and ship one small piece of work, however minor.", shape: "make_real", tier: "large", minutes: 45, builds: "Shipping, even small, is the habit that pays off.", unlockLevel: 5, subcategory: "Craftsmanship" },
    { id: "craft_end_to_end", action: "Get one feature working start to finish, even a thin one.", shape: "make_real", tier: "large", minutes: 60, builds: "A working slice end to end proves the whole thing can work.", unlockLevel: 5, subcategory: "Craftsmanship" },
    { id: "craft_remove_complexity", action: "Cut or simplify one part that's more complicated than it needs to be.", shape: "tidy_fix", tier: "medium", minutes: 30, builds: "Simpler is stronger, and cutting is harder than adding.", unlockLevel: 5, subcategory: "Craftsmanship" },
    { id: "craft_handle_edge", action: "Make one piece handle things going wrong without breaking.", shape: "make_real", tier: "medium", minutes: 35, builds: "Handling failure well is what makes a tool you can rely on.", unlockLevel: 5, subcategory: "Craftsmanship" },
    { id: "craft_measure", action: "Measure one real thing about your work instead of guessing.", shape: "discover", tier: "medium", minutes: 25, builds: "You can only improve what you actually measure.", unlockLevel: 5, subcategory: "Craftsmanship" },
    { id: "craft_polish_one", action: "Polish one detail until it's genuinely good, not just done.", shape: "make_real", tier: "medium", minutes: 30, builds: "Caring about one detail is how good taste grows.", unlockLevel: 5, subcategory: "Craftsmanship" },
    { id: "craft_reduce_friction", action: "Make one thing you do often easier to do.", shape: "tidy_fix", tier: "medium", minutes: 30, builds: "Smoothing a repeated job saves you time every future time.", unlockLevel: 5, subcategory: "Craftsmanship" },
    // --- Mastery (L10) ---
    { id: "craft_teach_build", action: "Show someone how to build a piece of what you make.", shape: "reach_out", tier: "large", minutes: 45, builds: "Teaching the craft makes you better at it too.", unlockLevel: 10, subcategory: "Mastery" },
    { id: "craft_design_system", action: "Build one small reusable thing instead of a one-off fix.", shape: "make_real", tier: "large", minutes: 60, builds: "Building something reusable saves work over and over.", unlockLevel: 10, subcategory: "Mastery" },
    { id: "craft_rebuild_better", action: "Rebuild something you made before, now that you know more.", shape: "make_real", tier: "large", minutes: 60, builds: "Redoing old work with new skill shows how far you've come.", unlockLevel: 10, subcategory: "Mastery" },
    { id: "craft_review_others", action: "Look over someone's work and give one useful note.", shape: "reach_out", tier: "medium", minutes: 30, builds: "Reviewing others' work sharpens your eye for your own.", unlockLevel: 10, subcategory: "Mastery" },
    { id: "craft_standard", action: "Set one rule that keeps future work consistent.", shape: "prepare", tier: "medium", minutes: 30, builds: "A good rule is a decision you only make once.", unlockLevel: 10, subcategory: "Mastery" },
    { id: "craft_mentor_session", action: "Spend a session helping someone get unstuck on their work.", shape: "reach_out", tier: "large", minutes: 45, builds: "Helping another builder is a sign you've come a long way.", unlockLevel: 10, subcategory: "Mastery" },
    // --- Legacy (L20) ---
    { id: "craft_finish_dormant", action: "Pick up and finish a project you gave up on.", shape: "make_real", tier: "large", minutes: 90, builds: "Closing an old project frees up energy you forgot it held.", unlockLevel: 20, subcategory: "Legacy" },
    { id: "craft_open_contribute", action: "Add one improvement to something bigger than your own work.", shape: "reach_out", tier: "large", minutes: 60, builds: "Building for others is the widest kind of craft.", unlockLevel: 20, subcategory: "Legacy" },
    { id: "craft_capstone", action: "Make real progress on the build that matters most to you.", shape: "make_real", tier: "large", minutes: 90, builds: "The work you care about most deserves your best hours.", unlockLevel: 20, subcategory: "Legacy" }
  ],
  creation: [
    // --- Foundations (L1) ---
    { id: "creation_one_artifact", action: "Make one small thing, however rough.", shape: "make_real", tier: "small", minutes: 15, builds: "One thing you made beats ten you only imagined.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "creation_sketch_first", action: "Sketch the idea before trying to make it good.", shape: "make_real", tier: "small", minutes: 12, builds: "A sketch lets the idea exist before you judge it.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "creation_one_line", action: "Make just one line, bar, or frame.", shape: "make_real", tier: "small", minutes: 10, builds: "One mark on the blank page makes the next one easy.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "creation_capture_idea", action: "Write down one idea before you forget it.", shape: "discover", tier: "small", minutes: 5, builds: "Saved ideas pile up into stuff you can use later.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "creation_warmup", action: "Do a 10-minute warm-up with no goal but to start.", shape: "make_real", tier: "small", minutes: 10, builds: "Warming up makes starting the real thing easier.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "creation_imitate", action: "Copy a small bit of something you like, to learn how it works.", shape: "consume", tier: "small", minutes: 20, builds: "Copying is how your hands learn what your eyes love.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "creation_gather", action: "Collect a few references for what you want to make.", shape: "discover", tier: "small", minutes: 15, builds: "References give your ideas something to push against.", unlockLevel: 1, subcategory: "Foundations" },
    // --- Practice (L2) ---
    { id: "creation_messy_draft", action: "Make the messiest first draft you can for 25 minutes.", shape: "make_real", tier: "medium", minutes: 25, builds: "At the start, momentum matters more than polish.", unlockLevel: 2, subcategory: "Practice" },
    { id: "creation_remix", action: "Take something that exists and turn it into something new.", shape: "discover", tier: "medium", minutes: 25, builds: "Mixing things together is where most new ideas come from.", unlockLevel: 2, subcategory: "Practice" },
    { id: "creation_constraint", action: "Make something with one tight rule you pick.", shape: "make_real", tier: "medium", minutes: 25, builds: "A limit frees you from too many choices.", unlockLevel: 2, subcategory: "Practice" },
    { id: "creation_finish_rough", action: "Take one rough piece all the way to 'good enough'.", shape: "make_real", tier: "medium", minutes: 30, builds: "Finishing roughly teaches more than starting perfectly.", unlockLevel: 2, subcategory: "Practice" },
    { id: "creation_daily_small", action: "Make one small thing today to keep a creative streak going.", shape: "make_real", tier: "small", minutes: 15, builds: "Making something small each day is how you become a maker.", unlockLevel: 2, subcategory: "Practice" },
    { id: "creation_study_master", action: "Look at how one thing you admire was actually made.", shape: "consume", tier: "medium", minutes: 25, builds: "Seeing how it's made turns magic into something you can learn.", unlockLevel: 2, subcategory: "Practice" },
    { id: "creation_new_medium", action: "Try making something in a way you don't usually use.", shape: "discover", tier: "medium", minutes: 30, builds: "A new medium stretches what you can imagine.", unlockLevel: 2, subcategory: "Practice" },
    // --- Voice (L5) ---
    { id: "creation_finish_piece", action: "Take one piece all the way to finished and shareable.", shape: "make_real", tier: "large", minutes: 50, builds: "A finished piece proves you can go the distance.", unlockLevel: 5, subcategory: "Voice" },
    { id: "creation_revise_deep", action: "Take one finished piece and make it clearly better.", shape: "make_real", tier: "medium", minutes: 35, builds: "Revising is where a draft turns into real work.", unlockLevel: 5, subcategory: "Voice" },
    { id: "creation_share_one", action: "Share one finished piece with at least one person.", shape: "reach_out", tier: "medium", minutes: 20, builds: "Sharing is what makes the making count.", unlockLevel: 5, subcategory: "Voice" },
    { id: "creation_seek_feedback", action: "Ask one person for honest feedback, and just listen.", shape: "reach_out", tier: "medium", minutes: 25, builds: "Honest feedback is the fastest way past your blind spots.", unlockLevel: 5, subcategory: "Voice" },
    { id: "creation_personal_voice", action: "Make one thing that only you would make, on purpose.", shape: "make_real", tier: "large", minutes: 45, builds: "Leaning into what's yours is how your style shows up.", unlockLevel: 5, subcategory: "Voice" },
    { id: "creation_kill_darling", action: "Cut the part you love but the work doesn't need.", shape: "tidy_fix", tier: "medium", minutes: 25, builds: "Putting the work first is a real step forward.", unlockLevel: 5, subcategory: "Voice" },
    { id: "creation_series", action: "Make a second piece that goes with one you already made.", shape: "make_real", tier: "medium", minutes: 40, builds: "A few pieces together turn one idea into a body of work.", unlockLevel: 5, subcategory: "Voice" },
    // --- Body of Work (L10) ---
    { id: "creation_publish", action: "Put one finished piece somewhere people can find it.", shape: "reach_out", tier: "large", minutes: 45, builds: "Putting it out there turns a maker into an artist.", unlockLevel: 10, subcategory: "Body of Work" },
    { id: "creation_collect", action: "Gather your pieces into one collection or portfolio.", shape: "make_real", tier: "large", minutes: 60, builds: "Seeing your work together shows the thread running through it.", unlockLevel: 10, subcategory: "Body of Work" },
    { id: "creation_commission", action: "Make something for a specific person or real purpose.", shape: "make_real", tier: "large", minutes: 60, builds: "Making for a real need sharpens your skills against reality.", unlockLevel: 10, subcategory: "Body of Work" },
    { id: "creation_teach_make", action: "Show someone how to make something you can make.", shape: "reach_out", tier: "medium", minutes: 35, builds: "Passing on the craft is a quiet sign you've mastered it.", unlockLevel: 10, subcategory: "Body of Work" },
    { id: "creation_critique_self", action: "Look honestly at one of your pieces and note what to work on.", shape: "discover", tier: "medium", minutes: 25, builds: "Seeing your own work clearly is what drives you forward.", unlockLevel: 10, subcategory: "Body of Work" },
    { id: "creation_collaborate", action: "Make one thing together with another person.", shape: "reach_out", tier: "large", minutes: 60, builds: "Working with others takes the work past what you'd reach alone.", unlockLevel: 10, subcategory: "Body of Work" },
    // --- Legacy (L20) ---
    { id: "creation_ambitious", action: "Make real progress on the most ambitious thing you want to create.", shape: "make_real", tier: "large", minutes: 90, builds: "The thing that scares you a little is the thing worth making.", unlockLevel: 20, subcategory: "Legacy" },
    { id: "creation_finish_dormant", action: "Finish a creative project you set aside long ago.", shape: "make_real", tier: "large", minutes: 90, builds: "Finishing the abandoned thing gives you a piece of yourself back.", unlockLevel: 20, subcategory: "Legacy" },
    { id: "creation_mentor_artist", action: "Help another maker finish or share something of theirs.", shape: "reach_out", tier: "large", minutes: 60, builds: "Lifting someone else's work is the most generous kind of craft.", unlockLevel: 20, subcategory: "Legacy" }
  ],
  mind: [
    // --- Foundations (L1) ---
    { id: "mind_relax", action: "Do one thing you find relaxing, with your full attention.", shape: "regulate", tier: "small", minutes: 15, builds: "A calm mind is the tool you do everything else with.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "mind_name_state", action: "Name how you're feeling in plain words before you do anything.", shape: "regulate", tier: "small", minutes: 5, builds: "Naming a feeling loosens its hold on your next move.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "mind_breathe", action: "Take ten slow breaths, counting each one.", shape: "regulate", tier: "small", minutes: 5, builds: "A few slow breaths reset you enough to choose well.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "mind_reflect", action: "Spend 10 minutes on what actually went well today.", shape: "consume", tier: "small", minutes: 10, builds: "Noticing what worked trains you toward more of it.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "mind_step_away", action: "Step away from the screen and look at something far away.", shape: "regulate", tier: "small", minutes: 5, builds: "A small break stops focus from slowly draining you.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "mind_one_kind", action: "Say one kind, true thing to yourself, like you would to a friend.", shape: "regulate", tier: "small", minutes: 3, builds: "How you talk to yourself sets the mood of your day.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "mind_water_light", action: "Get a glass of water and a few minutes of daylight.", shape: "do_a_rep", tier: "small", minutes: 5, builds: "The basics steady your mind more than you'd think.", unlockLevel: 1, subcategory: "Foundations" },
    // --- Clarity (L2) ---
    { id: "mind_brain_dump", action: "Write down everything on your mind for 15 minutes.", shape: "prepare", tier: "medium", minutes: 15, builds: "Getting it out of your head onto paper clears space to think.", unlockLevel: 2, subcategory: "Clarity" },
    { id: "mind_single_task", action: "Pick one task and do only it, with everything else closed.", shape: "regulate", tier: "medium", minutes: 25, builds: "Doing one thing at a time gets easier the more you practice.", unlockLevel: 2, subcategory: "Clarity" },
    { id: "mind_worry_window", action: "Give yourself 10 minutes to worry on purpose, then stop.", shape: "regulate", tier: "small", minutes: 10, builds: "Keeping worry to a set time stops it spreading everywhere.", unlockLevel: 2, subcategory: "Clarity" },
    { id: "mind_one_decision", action: "Make one small decision you've been putting off, and move on.", shape: "prepare", tier: "small", minutes: 10, builds: "Each thing you decide quietly gives you your focus back.", unlockLevel: 2, subcategory: "Clarity" },
    { id: "mind_reframe", action: "Write one stuck thought, then a truer, kinder way to see it.", shape: "regulate", tier: "medium", minutes: 15, builds: "Seeing a thought differently changes what feels possible.", unlockLevel: 2, subcategory: "Clarity" },
    { id: "mind_gratitude", action: "Write down three things you're glad about right now.", shape: "consume", tier: "small", minutes: 8, builds: "Naming what's good pulls your attention toward it.", unlockLevel: 2, subcategory: "Clarity" },
    { id: "mind_screen_fast", action: "Take a 20-minute break from all screens.", shape: "regulate", tier: "medium", minutes: 20, builds: "A short break from screens lets a busy mind settle.", unlockLevel: 2, subcategory: "Clarity" },
    // --- Steadiness (L5) ---
    { id: "mind_meditate", action: "Sit quietly and follow your breath for 15 minutes.", shape: "regulate", tier: "medium", minutes: 15, builds: "Bringing your attention back, again and again, is the practice.", unlockLevel: 5, subcategory: "Steadiness" },
    { id: "mind_journal_deep", action: "Write honestly about one thing that's been weighing on you.", shape: "prepare", tier: "medium", minutes: 25, builds: "Writing it out often settles what going in circles can't.", unlockLevel: 5, subcategory: "Steadiness" },
    { id: "mind_name_trigger", action: "Notice one thing that throws you off, and name it.", shape: "discover", tier: "medium", minutes: 15, builds: "A trigger you can name has less power to surprise you.", unlockLevel: 5, subcategory: "Steadiness" },
    { id: "mind_let_go", action: "Pick one thing you can't control and let it go on purpose.", shape: "regulate", tier: "small", minutes: 10, builds: "Letting go of what isn't yours gives you your energy back.", unlockLevel: 5, subcategory: "Steadiness" },
    { id: "mind_savor", action: "Do one ordinary thing slowly and really pay attention.", shape: "regulate", tier: "small", minutes: 10, builds: "Being present in small moments adds up to a calmer life.", unlockLevel: 5, subcategory: "Steadiness" },
    { id: "mind_digital_boundary", action: "Set one limit on a screen habit that scatters your focus.", shape: "prepare", tier: "medium", minutes: 15, builds: "Protecting your attention protects how clearly you think.", unlockLevel: 5, subcategory: "Steadiness" },
    { id: "mind_review_week", action: "Look back over the week and note one thing worth keeping.", shape: "consume", tier: "medium", minutes: 20, builds: "Seeing your own patterns is the first step to steering them.", unlockLevel: 5, subcategory: "Steadiness" },
    // --- Depth (L10) ---
    { id: "mind_values", action: "Write down what matters to you, then do one thing that fits it.", shape: "prepare", tier: "medium", minutes: 30, builds: "Acting on what matters is what makes a life feel like yours.", unlockLevel: 10, subcategory: "Depth" },
    { id: "mind_long_reflect", action: "Think about where you were a year ago versus now.", shape: "consume", tier: "medium", minutes: 30, builds: "The longer view shows growth the day-to-day hides.", unlockLevel: 10, subcategory: "Depth" },
    { id: "mind_forgive", action: "Work on letting go of one grudge you've been holding.", shape: "regulate", tier: "medium", minutes: 25, builds: "Letting go of an old hurt frees up energy stuck in it.", unlockLevel: 10, subcategory: "Depth" },
    { id: "mind_sit_uncertainty", action: "Sit with one open question without rushing to answer it.", shape: "regulate", tier: "medium", minutes: 20, builds: "Being okay with not knowing is a deeper kind of calm.", unlockLevel: 10, subcategory: "Depth" },
    { id: "mind_help_calm", action: "Help someone else steady themselves through something hard.", shape: "reach_out", tier: "medium", minutes: 30, builds: "Offering calm to someone else deepens your own.", unlockLevel: 10, subcategory: "Depth" },
    { id: "mind_design_routine", action: "Set up one small daily routine that protects your mind.", shape: "prepare", tier: "medium", minutes: 25, builds: "A routine you trust saves you from deciding every time.", unlockLevel: 10, subcategory: "Depth" },
    // --- Equanimity (L20) ---
    { id: "mind_face_fear", action: "Think honestly about one fear that quietly shapes your choices.", shape: "discover", tier: "large", minutes: 40, builds: "Seeing a hidden fear clearly is how it stops steering you.", unlockLevel: 20, subcategory: "Equanimity" },
    { id: "mind_meaning", action: "Think about what makes your effort feel worth it, and write it down.", shape: "prepare", tier: "large", minutes: 45, builds: "Knowing your why steadies you through the hard parts.", unlockLevel: 20, subcategory: "Equanimity" },
    { id: "mind_teach_practice", action: "Share one thing that steadies you with someone who needs it.", shape: "reach_out", tier: "medium", minutes: 30, builds: "Passing on what helps you helps it take deeper root.", unlockLevel: 20, subcategory: "Equanimity" }
  ],
  learning: [
    // --- Foundations (L1) ---
    { id: "learn_one_chapter", action: "Read at least one chapter or section of something.", shape: "consume", tier: "small", minutes: 20, builds: "One piece you understand becomes the base for the next.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "learn_explain_aloud", action: "Say something you're learning out loud, in your own words.", shape: "make_real", tier: "small", minutes: 10, builds: "If you can say it simply, you really understand it.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "learn_one_example", action: "Find the one example that makes a hard idea click.", shape: "discover", tier: "small", minutes: 15, builds: "The right example unlocks the whole idea.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "learn_one_word", action: "Look up one thing you didn't get and write down the answer.", shape: "discover", tier: "small", minutes: 8, builds: "Closing one small gap keeps the big picture clear.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "learn_question", action: "Write down one real question you have about a topic.", shape: "prepare", tier: "small", minutes: 8, builds: "A good question is the hook learning hangs on.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "learn_five_min", action: "Study one thing for just 10 minutes, then stop if you want.", shape: "consume", tier: "small", minutes: 10, builds: "A little most days beats a lot once in a while.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "learn_note_one", action: "Write one clear note in your own words on what you learned.", shape: "make_real", tier: "small", minutes: 10, builds: "Notes in your own words are notes you'll actually use.", unlockLevel: 1, subcategory: "Foundations" },
    // --- Study (L2) ---
    { id: "learn_documentary", action: "Watch one documentary or talk on something you're curious about.", shape: "consume", tier: "medium", minutes: 45, builds: "Following your curiosity is how a subject opens up.", unlockLevel: 2, subcategory: "Study" },
    { id: "learn_recall", action: "Study one thing, then try to recall it without looking.", shape: "consume", tier: "medium", minutes: 20, builds: "Recalling, not rereading, is what makes it stick.", unlockLevel: 2, subcategory: "Study" },
    { id: "learn_summarize", action: "Sum up what you learned today in three sentences.", shape: "make_real", tier: "small", minutes: 12, builds: "Boiling an idea down forces you to find its core.", unlockLevel: 2, subcategory: "Study" },
    { id: "learn_practice_problem", action: "Work through one practice problem on your own.", shape: "make_real", tier: "medium", minutes: 25, builds: "Doing beats watching; the struggle is where it sinks in.", unlockLevel: 2, subcategory: "Study" },
    { id: "learn_connect", action: "Link one new idea to something you already know.", shape: "discover", tier: "small", minutes: 15, builds: "Ideas you connect are ideas you can find again.", unlockLevel: 2, subcategory: "Study" },
    { id: "learn_flashcard", action: "Make a few review prompts for something you want to remember.", shape: "make_real", tier: "small", minutes: 15, builds: "Reviewing over time turns quick effort into lasting memory.", unlockLevel: 2, subcategory: "Study" },
    { id: "learn_listen", action: "Listen to a lesson or talk while you do chores.", shape: "consume", tier: "medium", minutes: 30, builds: "Spare time becomes learning time with no extra hours.", unlockLevel: 2, subcategory: "Study" },
    // --- Mastery (L5) ---
    { id: "learn_teach_someone", action: "Teach one thing you know to another person.", shape: "reach_out", tier: "medium", minutes: 20, builds: "Teaching shows you exactly where your understanding is thin.", unlockLevel: 5, subcategory: "Mastery" },
    { id: "learn_deliberate", action: "Spend 25 minutes practicing the part you're worst at.", shape: "make_real", tier: "medium", minutes: 25, builds: "Working on the weak spot is where real skill is won.", unlockLevel: 5, subcategory: "Mastery" },
    { id: "learn_first_principles", action: "Break one idea down to the basics and build it back up.", shape: "discover", tier: "medium", minutes: 30, builds: "Knowing why, not just what, is knowledge that lasts.", unlockLevel: 5, subcategory: "Mastery" },
    { id: "learn_apply_real", action: "Use something you learned in one real situation today.", shape: "make_real", tier: "medium", minutes: 30, builds: "Knowledge you use is knowledge that becomes yours.", unlockLevel: 5, subcategory: "Mastery" },
    { id: "learn_compare_sources", action: "Compare two sources on the same topic and note where they differ.", shape: "discover", tier: "medium", minutes: 30, builds: "Seeing them disagree gets you past surface understanding.", unlockLevel: 5, subcategory: "Mastery" },
    { id: "learn_test_self", action: "Quiz yourself on a topic and honestly mark what you missed.", shape: "discover", tier: "medium", minutes: 25, builds: "Finding the gaps is the quickest way to close them.", unlockLevel: 5, subcategory: "Mastery" },
    { id: "learn_project_learn", action: "Start a small project that makes you learn as you go.", shape: "make_real", tier: "large", minutes: 45, builds: "A real project teaches what reading alone can't.", unlockLevel: 5, subcategory: "Mastery" },
    // --- Teaching (L10) ---
    { id: "learn_write_explainer", action: "Write a short, clear explanation of something you know well.", shape: "make_real", tier: "large", minutes: 45, builds: "Writing to teach others sharpens your own grasp.", unlockLevel: 10, subcategory: "Teaching" },
    { id: "learn_answer_question", action: "Answer one real question someone has, as clearly as you can.", shape: "reach_out", tier: "medium", minutes: 20, builds: "Helping someone learn is learning, twice over.", unlockLevel: 10, subcategory: "Teaching" },
    { id: "learn_curate", action: "Pick the best resources on a topic for someone just starting.", shape: "make_real", tier: "medium", minutes: 30, builds: "Knowing what's worth learning is its own kind of skill.", unlockLevel: 10, subcategory: "Teaching" },
    { id: "learn_debate_idea", action: "Argue the best case against something you believe.", shape: "discover", tier: "medium", minutes: 30, builds: "Making the other side's case is how understanding grows up.", unlockLevel: 10, subcategory: "Teaching" },
    { id: "learn_study_group", action: "Learn one thing with another person, out loud.", shape: "reach_out", tier: "medium", minutes: 35, builds: "Learning together shows up both of your blind spots.", unlockLevel: 10, subcategory: "Teaching" },
    { id: "learn_synthesize", action: "Pull a few things you've learned into one bigger idea.", shape: "make_real", tier: "large", minutes: 40, builds: "Putting the pieces together is where insight shows up.", unlockLevel: 10, subcategory: "Teaching" },
    // --- Scholarship (L20) ---
    { id: "learn_deep_topic", action: "Go deep on one topic that really fascinates you.", shape: "consume", tier: "large", minutes: 90, builds: "Sticking with one thing is how real expertise grows.", unlockLevel: 20, subcategory: "Scholarship" },
    { id: "learn_original", action: "Work out and write down your own view on a topic.", shape: "make_real", tier: "large", minutes: 60, builds: "Having your own view is the far end of understanding.", unlockLevel: 20, subcategory: "Scholarship" },
    { id: "learn_mentor_learner", action: "Guide someone else through learning something hard.", shape: "reach_out", tier: "large", minutes: 60, builds: "Helping someone else learn deepens what you know.", unlockLevel: 20, subcategory: "Scholarship" }
  ],
  planning: [
    // --- Foundations (L1) ---
    { id: "plan_next_action", action: "Write down the next clear step for a stuck goal.", shape: "prepare", tier: "small", minutes: 8, builds: "A clear next step is the cure for feeling stuck.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "plan_map_step", action: "Map just the next step, not the whole path.", shape: "prepare", tier: "small", minutes: 12, builds: "You only ever need the next step to keep moving.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "plan_three_options", action: "List three options, then pick the smallest one to start now.", shape: "prepare", tier: "small", minutes: 10, builds: "Picking the smallest start breaks the stall.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "plan_top_three", action: "Write the three things that actually matter today.", shape: "prepare", tier: "small", minutes: 8, builds: "Naming the few that matter quiets the noise.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "plan_brain_sweep", action: "List everything you're trying to keep track of, in one place.", shape: "prepare", tier: "small", minutes: 12, builds: "A written list is more trustworthy than a tired memory.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "plan_one_deadline", action: "Give one floating task a real date and time.", shape: "prepare", tier: "small", minutes: 8, builds: "A date turns a someday into something that happens.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "plan_tomorrow", action: "Decide tonight what your first move tomorrow will be.", shape: "prepare", tier: "small", minutes: 8, builds: "Deciding ahead saves tomorrow's energy for doing.", unlockLevel: 1, subcategory: "Foundations" },
    // --- Mapping (L2) ---
    { id: "plan_obstacle", action: "Write down the main thing in your way, then one way around it.", shape: "prepare", tier: "medium", minutes: 15, builds: "Naming the problem turns vague dread into something you can solve.", unlockLevel: 2, subcategory: "Mapping" },
    { id: "plan_weekly_focus", action: "Pick the one thing that matters most this week and write it down.", shape: "prepare", tier: "medium", minutes: 15, builds: "One clear focus keeps your effort from scattering.", unlockLevel: 2, subcategory: "Mapping" },
    { id: "plan_break_down", action: "Break one big task into three smaller ones.", shape: "prepare", tier: "medium", minutes: 15, builds: "A big thing in small steps stops being scary.", unlockLevel: 2, subcategory: "Mapping" },
    { id: "plan_estimate", action: "Guess how long one task will really take, then check later.", shape: "discover", tier: "small", minutes: 10, builds: "Honest guesses are how plans stop lying to you.", unlockLevel: 2, subcategory: "Mapping" },
    { id: "plan_calendar_block", action: "Block real time on your calendar for one important task.", shape: "prepare", tier: "small", minutes: 10, builds: "Time you've claimed is time the day can't steal.", unlockLevel: 2, subcategory: "Mapping" },
    { id: "plan_dependencies", action: "Note what one task is waiting on before it can start.", shape: "prepare", tier: "medium", minutes: 15, builds: "Spotting what blocks it saves you pushing a locked door.", unlockLevel: 2, subcategory: "Mapping" },
    { id: "plan_say_no", action: "Drop or decline one thing that isn't really a priority.", shape: "prepare", tier: "small", minutes: 10, builds: "Every honest no protects time for a real yes.", unlockLevel: 2, subcategory: "Mapping" },
    // --- Strategy (L5) ---
    { id: "plan_milestones", action: "Break one goal into a few dated checkpoints.", shape: "prepare", tier: "medium", minutes: 25, builds: "Checkpoints turn a far-off goal into a set of near wins.", unlockLevel: 5, subcategory: "Strategy" },
    { id: "plan_premortem", action: "Imagine the plan failed, write why, and prevent one cause.", shape: "discover", tier: "medium", minutes: 25, builds: "Seeing failure coming is the cheapest way to avoid it.", unlockLevel: 5, subcategory: "Strategy" },
    { id: "plan_prioritize", action: "Rank your tasks by impact and do the top one first.", shape: "prepare", tier: "medium", minutes: 20, builds: "Doing the most important thing first beats doing the most things.", unlockLevel: 5, subcategory: "Strategy" },
    { id: "plan_constraints", action: "Write down the real limits on a goal — time, energy, money.", shape: "prepare", tier: "medium", minutes: 20, builds: "A plan that respects your limits is one that survives.", unlockLevel: 5, subcategory: "Strategy" },
    { id: "plan_if_then", action: "Write one if-this-then-that plan for a moment you usually slip.", shape: "prepare", tier: "medium", minutes: 15, builds: "Deciding ahead of time beats willpower in the moment.", unlockLevel: 5, subcategory: "Strategy" },
    { id: "plan_tradeoff", action: "Name one real trade-off you're making and choose it on purpose.", shape: "discover", tier: "medium", minutes: 20, builds: "Choosing your trade-offs beats having them chosen for you.", unlockLevel: 5, subcategory: "Strategy" },
    { id: "plan_buffer", action: "Add some extra time to a plan that's been running tight.", shape: "prepare", tier: "small", minutes: 12, builds: "Slack in a plan keeps one delay from breaking it.", unlockLevel: 5, subcategory: "Strategy" },
    // --- Systems (L10) ---
    { id: "plan_weekly_review", action: "Review last week and plan the next in one sitting.", shape: "prepare", tier: "large", minutes: 45, builds: "A weekly review is the steering wheel of a busy life.", unlockLevel: 10, subcategory: "Systems" },
    { id: "plan_recurring", action: "Turn one repeated choice into a set rule or routine.", shape: "prepare", tier: "medium", minutes: 25, builds: "A good system makes the right choice the easy one.", unlockLevel: 10, subcategory: "Systems" },
    { id: "plan_template", action: "Make a reusable checklist for something you do often.", shape: "make_real", tier: "medium", minutes: 30, builds: "A checklist made once saves you from forgetting forever.", unlockLevel: 10, subcategory: "Systems" },
    { id: "plan_metrics", action: "Pick one number that tells you if a goal is on track.", shape: "discover", tier: "medium", minutes: 25, builds: "The right number turns a vague hope into something you can steer.", unlockLevel: 10, subcategory: "Systems" },
    { id: "plan_delegate", action: "Hand off or automate one thing that doesn't need you.", shape: "prepare", tier: "medium", minutes: 25, builds: "Letting go of the extra frees you for what only you can do.", unlockLevel: 10, subcategory: "Systems" },
    { id: "plan_review_system", action: "Fix one planning habit that keeps letting you down.", shape: "tidy_fix", tier: "medium", minutes: 30, builds: "Fixing the system beats trying harder inside a broken one.", unlockLevel: 10, subcategory: "Systems" },
    // --- Vision (L20) ---
    { id: "plan_quarter", action: "Set a few real goals for the season ahead.", shape: "prepare", tier: "large", minutes: 60, builds: "A clear season keeps your daily effort pointed somewhere.", unlockLevel: 20, subcategory: "Vision" },
    { id: "plan_values_align", action: "Check whether your plans actually serve what you care about most.", shape: "discover", tier: "large", minutes: 45, builds: "Plans that fit your values are the ones worth finishing.", unlockLevel: 20, subcategory: "Vision" },
    { id: "plan_long_horizon", action: "Sketch where you want to be in a few years, in plain words.", shape: "prepare", tier: "large", minutes: 60, builds: "A far-off picture quietly shapes every nearer choice.", unlockLevel: 20, subcategory: "Vision" }
  ],
  order: [
    // --- Foundations (L1) ---
    { id: "order_tidy", action: "Tidy one small area for 10 minutes.", shape: "tidy_fix", tier: "small", minutes: 10, builds: "A clear space quietly clears the mind that works in it.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "order_one_item", action: "Finish one thing on your list before anything else.", shape: "tidy_fix", tier: "small", minutes: 15, builds: "One thing done beats five things half-started.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "order_one_surface", action: "Clear one surface completely — a desk, a counter, a screen.", shape: "tidy_fix", tier: "small", minutes: 10, builds: "One clear surface is a small patch of calm to work from.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "order_put_back", action: "Put five things back where they belong.", shape: "tidy_fix", tier: "small", minutes: 8, builds: "A few things put away keeps small messes from piling up.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "order_two_min", action: "Do one task that takes under two minutes, right now.", shape: "do_a_rep", tier: "small", minutes: 5, builds: "Quick tasks done now never become tomorrow's pile.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "order_trash", action: "Throw away or recycle one thing you don't need.", shape: "tidy_fix", tier: "small", minutes: 5, builds: "Letting one thing go makes room, in more ways than one.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "order_inbox_five", action: "Deal with five items in one full inbox or queue.", shape: "tidy_fix", tier: "small", minutes: 12, builds: "Chipping at the pile keeps it from becoming a wall.", unlockLevel: 1, subcategory: "Foundations" },
    // --- Maintenance (L2) ---
    { id: "order_unblock", action: "Do the one small task that unblocks the most other things.", shape: "prepare", tier: "medium", minutes: 20, builds: "Clearing a bottleneck frees everything behind it.", unlockLevel: 2, subcategory: "Maintenance" },
    { id: "order_reset", action: "Reset your desk or inbox back to a calm starting point.", shape: "tidy_fix", tier: "medium", minutes: 20, builds: "A clean starting point makes the next start easy.", unlockLevel: 2, subcategory: "Maintenance" },
    { id: "order_one_room", action: "Get one room back in order for 20 minutes.", shape: "tidy_fix", tier: "medium", minutes: 20, builds: "An ordered space makes everything you do there easier.", unlockLevel: 2, subcategory: "Maintenance" },
    { id: "order_overdue", action: "Handle one small overdue task you keep avoiding.", shape: "do_a_rep", tier: "small", minutes: 15, builds: "Clearing a nagging task gives back the attention it ate.", unlockLevel: 2, subcategory: "Maintenance" },
    { id: "order_files", action: "Sort out one messy folder, drawer, or set of files.", shape: "tidy_fix", tier: "medium", minutes: 20, builds: "Things you can find are things you can use.", unlockLevel: 2, subcategory: "Maintenance" },
    { id: "order_errand", action: "Knock out one errand or admin task you've been putting off.", shape: "do_a_rep", tier: "medium", minutes: 20, builds: "Each finished errand is one less thing nagging at you.", unlockLevel: 2, subcategory: "Maintenance" },
    { id: "order_prep_tomorrow", action: "Set out what you'll need for tomorrow tonight.", shape: "prepare", tier: "small", minutes: 10, builds: "A ready morning starts with momentum instead of friction.", unlockLevel: 2, subcategory: "Maintenance" },
    // --- Systems (L5) ---
    { id: "order_fix_system", action: "Fix one small routine that keeps failing you.", shape: "tidy_fix", tier: "large", minutes: 30, builds: "Fixing the routine once saves the same effort many times.", unlockLevel: 5, subcategory: "Systems" },
    { id: "order_home_one", action: "Give one homeless pile of stuff a permanent spot.", shape: "tidy_fix", tier: "medium", minutes: 25, builds: "A spot for everything is what keeps tidy easy.", unlockLevel: 5, subcategory: "Systems" },
    { id: "order_routine", action: "Set up one small daily routine that keeps things in order.", shape: "prepare", tier: "medium", minutes: 25, builds: "A routine keeps order without spending fresh willpower.", unlockLevel: 5, subcategory: "Systems" },
    { id: "order_label", action: "Label or sort one thing so it stays organized on its own.", shape: "make_real", tier: "small", minutes: 15, builds: "A little structure now saves a lot of searching later.", unlockLevel: 5, subcategory: "Systems" },
    { id: "order_declutter_zone", action: "Clear out one area, deciding keep, donate, or toss for each thing.", shape: "tidy_fix", tier: "large", minutes: 40, builds: "Less stuff means less to manage, clean, and think about.", unlockLevel: 5, subcategory: "Systems" },
    { id: "order_digital", action: "Clean up one corner of your digital space.", shape: "tidy_fix", tier: "medium", minutes: 25, builds: "Digital order is as restful as a tidy room.", unlockLevel: 5, subcategory: "Systems" },
    { id: "order_batch", action: "Group a few similar small tasks and do them together.", shape: "do_a_rep", tier: "medium", minutes: 25, builds: "Doing like with like cuts the cost of switching.", unlockLevel: 5, subcategory: "Systems" },
    // --- Optimization (L10) ---
    { id: "order_audit", action: "Find where your time or stuff leaks, and plug one hole.", shape: "discover", tier: "medium", minutes: 30, builds: "Finding the leak is most of the work of fixing it.", unlockLevel: 10, subcategory: "Optimization" },
    { id: "order_automate", action: "Automate one task you keep doing by hand.", shape: "make_real", tier: "medium", minutes: 35, builds: "A task automated once is effort you never spend again.", unlockLevel: 10, subcategory: "Optimization" },
    { id: "order_streamline", action: "Cut one process down to fewer steps.", shape: "tidy_fix", tier: "medium", minutes: 30, builds: "Fewer steps means fewer places to go wrong.", unlockLevel: 10, subcategory: "Optimization" },
    { id: "order_maintenance_plan", action: "Set a repeating reminder for one thing you always forget.", shape: "prepare", tier: "small", minutes: 12, builds: "A reminder outlasts the best intentions.", unlockLevel: 10, subcategory: "Optimization" },
    { id: "order_review_commitments", action: "Look over your commitments and let one stale one go.", shape: "discover", tier: "medium", minutes: 30, builds: "Trimming what you've taken on keeps your yes meaningful.", unlockLevel: 10, subcategory: "Optimization" },
    { id: "order_one_in_out", action: "Try one-in-one-out for a pile that keeps growing.", shape: "prepare", tier: "small", minutes: 12, builds: "A simple rule holds back the slow creep of clutter.", unlockLevel: 10, subcategory: "Optimization" },
    // --- Stewardship (L20) ---
    { id: "order_big_declutter", action: "Tackle one space you've been avoiding and get it fully in order.", shape: "tidy_fix", tier: "large", minutes: 90, builds: "Reclaiming the dreaded space gives back the energy it drained.", unlockLevel: 20, subcategory: "Stewardship" },
    { id: "order_help_organize", action: "Help someone else get a corner of their life in order.", shape: "reach_out", tier: "large", minutes: 60, builds: "Order shared is order doubled.", unlockLevel: 20, subcategory: "Stewardship" },
    { id: "order_overhaul", action: "Rebuild one system in your life that's outgrown itself.", shape: "make_real", tier: "large", minutes: 90, builds: "A system built for who you are now removes daily friction.", unlockLevel: 20, subcategory: "Stewardship" }
  ],
  body: [
    // CAPABILITY / BEHAVIOR ONLY. Never weight/fat/aesthetic targets, never restriction.
    // --- Foundations (L1) ---
    { id: "body_move", action: "Move your body for 15 minutes at an easy pace.", shape: "do_a_rep", tier: "small", minutes: 15, builds: "Movement is a deposit into how you'll feel tomorrow.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "body_wake_clean", action: "Get up with your alarm without hitting snooze, once.", shape: "do_a_rep", tier: "small", minutes: 1, builds: "A clean start sets up everything after it.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "body_walk", action: "Take a 15-minute walk, no destination needed.", shape: "go_explore", tier: "small", minutes: 15, builds: "A walk clears the body and the head at once.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "body_hydrate_fuel", action: "Add one good thing to a meal as fuel, not as a rule.", shape: "do_a_rep", tier: "small", minutes: 5, builds: "Feeding yourself well is care, not control.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "body_stretch", action: "Stretch gently for 8 minutes, wherever feels tight.", shape: "do_a_rep", tier: "small", minutes: 8, builds: "A little stretching keeps the body comfortable to live in.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "body_stand", action: "Stand up and move around after sitting too long.", shape: "do_a_rep", tier: "small", minutes: 5, builds: "Breaking up sitting protects your energy all day.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "body_water", action: "Drink a glass of water now.", shape: "do_a_rep", tier: "small", minutes: 2, builds: "Staying hydrated is the cheapest way to feel better.", unlockLevel: 1, subcategory: "Foundations" },
    // --- Movement (L2) ---
    { id: "body_corrective", action: "Do one corrective movement for 15 minutes, slow and steady.", shape: "do_a_rep", tier: "medium", minutes: 15, builds: "Balanced movement keeps the body able and steady.", unlockLevel: 2, subcategory: "Movement" },
    { id: "body_workout_short", action: "Do a short workout you can comfortably repeat.", shape: "do_a_rep", tier: "medium", minutes: 20, builds: "Repeatable beats intense; showing up is what builds you.", unlockLevel: 2, subcategory: "Movement" },
    { id: "body_strength_basic", action: "Do a few sets of one bodyweight move, within comfort.", shape: "do_a_rep", tier: "medium", minutes: 15, builds: "Strength built gently is strength that sticks around.", unlockLevel: 2, subcategory: "Movement" },
    { id: "body_active_commute", action: "Make one trip today on foot or by bike instead.", shape: "go_explore", tier: "medium", minutes: 25, builds: "Built-in movement adds up without adding time.", unlockLevel: 2, subcategory: "Movement" },
    { id: "body_play", action: "Do one physical thing that's actually fun for you.", shape: "do_a_rep", tier: "medium", minutes: 30, builds: "Movement you enjoy is movement you'll keep doing.", unlockLevel: 2, subcategory: "Movement" },
    { id: "body_posture", action: "Fix your posture and hold it comfortably for a while.", shape: "regulate", tier: "small", minutes: 10, builds: "Easy posture lightens the load your body carries all day.", unlockLevel: 2, subcategory: "Movement" },
    { id: "body_mobility_routine", action: "Run through a short mobility routine for stiff spots.", shape: "do_a_rep", tier: "medium", minutes: 15, builds: "Mobility work keeps you moving freely for years.", unlockLevel: 2, subcategory: "Movement" },
    // --- Conditioning (L5) ---
    { id: "body_progress_movement", action: "Do a workout a little harder than last time, within comfort.", shape: "do_a_rep", tier: "medium", minutes: 30, builds: "Small steps up are how you get stronger without getting hurt.", unlockLevel: 5, subcategory: "Conditioning" },
    { id: "body_endurance", action: "Keep up easy movement a bit longer than usual today.", shape: "do_a_rep", tier: "medium", minutes: 35, builds: "Stamina built slowly widens what your days can hold.", unlockLevel: 5, subcategory: "Conditioning" },
    { id: "body_skill_practice", action: "Practice one physical skill you want to get better at.", shape: "do_a_rep", tier: "medium", minutes: 30, builds: "A body that learns new skills stays curious and able.", unlockLevel: 5, subcategory: "Conditioning" },
    { id: "body_balance", action: "Practice balance or coordination for a few minutes.", shape: "do_a_rep", tier: "small", minutes: 10, builds: "Balance is the quiet skill that protects you for life.", unlockLevel: 5, subcategory: "Conditioning" },
    { id: "body_cook_nourish", action: "Cook one good meal for yourself, as a way of caring for you.", shape: "make_real", tier: "medium", minutes: 35, builds: "Feeding yourself well is a daily bit of self-respect.", unlockLevel: 5, subcategory: "Conditioning" },
    { id: "body_sleep_routine", action: "Follow a calm wind-down routine before bed tonight.", shape: "regulate", tier: "medium", minutes: 30, builds: "A real wind-down is how good sleep, and good days, are made.", unlockLevel: 5, subcategory: "Conditioning" },
    { id: "body_outdoors_active", action: "Do something active outdoors for 30 minutes.", shape: "go_explore", tier: "medium", minutes: 30, builds: "Moving outdoors lifts the body and the mood together.", unlockLevel: 5, subcategory: "Conditioning" },
    // --- Resilience (L10) ---
    { id: "body_consistency_week", action: "Move your body on most days this week, gently.", shape: "do_a_rep", tier: "large", minutes: 30, builds: "A steady week beats one hard day by a long way.", unlockLevel: 10, subcategory: "Resilience" },
    { id: "body_address_weakness", action: "Work on one area your body keeps telling you needs attention.", shape: "do_a_rep", tier: "medium", minutes: 25, builds: "Tending a weak spot keeps the whole body steady.", unlockLevel: 10, subcategory: "Resilience" },
    { id: "body_recovery_active", action: "Do gentle, easy movement the day after a hard effort.", shape: "regulate", tier: "small", minutes: 15, builds: "Recovery is where the body actually gets stronger.", unlockLevel: 10, subcategory: "Resilience" },
    { id: "body_checkup", action: "Book or go to one health check you've been putting off.", shape: "prepare", tier: "medium", minutes: 30, builds: "Looking after the body is how it keeps looking after you.", unlockLevel: 10, subcategory: "Resilience" },
    { id: "body_listen", action: "Notice one thing your body's telling you and respond kindly.", shape: "discover", tier: "small", minutes: 10, builds: "Listening to your body builds trust that keeps you safe.", unlockLevel: 10, subcategory: "Resilience" },
    { id: "body_habit_swap", action: "Swap one draining habit for one that gives you energy, once.", shape: "do_a_rep", tier: "medium", minutes: 20, builds: "Trading up one habit lifts how you feel overall.", unlockLevel: 10, subcategory: "Resilience" },
    // --- Vitality (L20) ---
    { id: "body_capability_goal", action: "Take one real step toward a physical thing you want to be able to do.", shape: "do_a_rep", tier: "large", minutes: 45, builds: "Training toward a real ability gives movement a point.", unlockLevel: 20, subcategory: "Vitality" },
    { id: "body_move_with_others", action: "Be active with other people, once.", shape: "reach_out", tier: "large", minutes: 45, builds: "Moving with others makes it social and easier to keep up.", unlockLevel: 20, subcategory: "Vitality" },
    { id: "body_long_term_care", action: "Set up one lasting habit that protects your long-term health.", shape: "prepare", tier: "large", minutes: 45, builds: "The body you'll have in years is built by today's small care.", unlockLevel: 20, subcategory: "Vitality" }
  ],
  recovery: [
    // --- Foundations (L1) ---
    { id: "recovery_action", action: "Do one restful thing with no multitasking at all.", shape: "regulate", tier: "small", minutes: 15, builds: "Rest done on purpose is what makes the next push possible.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "recovery_nature", action: "Sit somewhere calm and do nothing for 10 minutes.", shape: "regulate", tier: "small", minutes: 10, builds: "Sitting still is a skill, and it gets easier.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "recovery_pause", action: "Take one real pause before the next thing.", shape: "regulate", tier: "small", minutes: 5, builds: "A small pause keeps the day from running you.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "recovery_unclench", action: "Notice and relax your jaw, shoulders, or hands.", shape: "regulate", tier: "small", minutes: 3, builds: "Letting go of held tension is rest you can take anywhere.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "recovery_break", action: "Take one real break away from work for 10 minutes.", shape: "regulate", tier: "small", minutes: 10, builds: "A real break refills you more than pushing through does.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "recovery_comfort", action: "Do one small comforting thing for yourself.", shape: "regulate", tier: "small", minutes: 10, builds: "Small kindnesses to yourself keep you from running empty.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "recovery_fresh_air", action: "Step outside for a few minutes of fresh air.", shape: "go_explore", tier: "small", minutes: 5, builds: "A change of air resets more than its small cost.", unlockLevel: 1, subcategory: "Foundations" },
    // --- Rest (L2) ---
    { id: "recovery_reduce_stim", action: "Turn off one source of noise or input for 20 minutes.", shape: "regulate", tier: "small", minutes: 20, builds: "Less coming in lets a frazzled mind settle.", unlockLevel: 2, subcategory: "Rest" },
    { id: "recovery_rest_protect", action: "Rest in a way that sets up your next move, guilt-free.", shape: "regulate", tier: "medium", minutes: 30, builds: "Rest counts as progress here, not as quitting.", unlockLevel: 2, subcategory: "Rest" },
    { id: "recovery_nap", action: "Take a short rest or nap when you really need one.", shape: "regulate", tier: "small", minutes: 20, builds: "A well-timed rest can save a whole afternoon.", unlockLevel: 2, subcategory: "Rest" },
    { id: "recovery_screen_off", action: "Put screens away for 20 minutes and rest your eyes.", shape: "regulate", tier: "small", minutes: 20, builds: "Resting your eyes and attention is real rest, not slacking.", unlockLevel: 2, subcategory: "Rest" },
    { id: "recovery_slow_meal", action: "Eat one meal slowly, without working or scrolling.", shape: "regulate", tier: "small", minutes: 20, builds: "An unhurried meal is a small daily bit of rest.", unlockLevel: 2, subcategory: "Rest" },
    { id: "recovery_bath", action: "Do one slow, soothing thing — a bath, tea, a warm shower.", shape: "regulate", tier: "small", minutes: 20, builds: "A soothing routine tells your body it's safe to settle.", unlockLevel: 2, subcategory: "Rest" },
    { id: "recovery_music", action: "Listen to something calming, really listening, for 15 minutes.", shape: "regulate", tier: "small", minutes: 15, builds: "Letting calm sound wash over you lowers your whole baseline.", unlockLevel: 2, subcategory: "Rest" },
    // --- Boundaries (L5) ---
    { id: "recovery_boundary", action: "Set one small limit that protects your energy today.", shape: "prepare", tier: "medium", minutes: 10, builds: "A protected limit keeps your reserves from draining.", unlockLevel: 5, subcategory: "Boundaries" },
    { id: "recovery_say_no", action: "Say no to one thing that would drain energy you don't have.", shape: "reach_out", tier: "small", minutes: 10, builds: "A kind no protects the energy a real yes needs.", unlockLevel: 5, subcategory: "Boundaries" },
    { id: "recovery_unplug_window", action: "Set one stretch of today where you're unreachable on purpose.", shape: "prepare", tier: "medium", minutes: 30, builds: "Protected quiet is where a worn-out mind refills.", unlockLevel: 5, subcategory: "Boundaries" },
    { id: "recovery_offload", action: "Hand off or put off one thing that's overloading you now.", shape: "prepare", tier: "small", minutes: 15, builds: "Dropping one load keeps the whole stack from tipping over.", unlockLevel: 5, subcategory: "Boundaries" },
    { id: "recovery_no_overtime", action: "Stop working at a reasonable time today, on purpose.", shape: "regulate", tier: "small", minutes: 5, builds: "An honest stopping point keeps tomorrow doable.", unlockLevel: 5, subcategory: "Boundaries" },
    { id: "recovery_protect_sleep", action: "Protect your sleep tonight from one thing that usually steals it.", shape: "prepare", tier: "small", minutes: 10, builds: "Guarded sleep is the biggest lever on how you recover.", unlockLevel: 5, subcategory: "Boundaries" },
    { id: "recovery_ask_help", action: "Ask for help with one thing instead of carrying it alone.", shape: "reach_out", tier: "small", minutes: 15, builds: "Letting others share the load is a skill, not a weakness.", unlockLevel: 5, subcategory: "Boundaries" },
    // --- Restoration (L10) ---
    { id: "recovery_full_rest_day", action: "Give yourself one real stretch of rest with no to-dos.", shape: "regulate", tier: "large", minutes: 60, builds: "Deep rest is what keeps steady effort possible.", unlockLevel: 10, subcategory: "Restoration" },
    { id: "recovery_refill", action: "Do one thing that really refills you, fully and unhurried.", shape: "regulate", tier: "medium", minutes: 40, builds: "Knowing what refills you, and doing it, is real self-care.", unlockLevel: 10, subcategory: "Restoration" },
    { id: "recovery_disconnect", action: "Take a longer break from a draining place or input.", shape: "regulate", tier: "medium", minutes: 45, builds: "Stepping back from the drain is how perspective comes back.", unlockLevel: 10, subcategory: "Restoration" },
    { id: "recovery_reflect_energy", action: "Notice what's draining you most, and name one change.", shape: "discover", tier: "medium", minutes: 25, builds: "Spotting the real drain is the first step to stopping it.", unlockLevel: 10, subcategory: "Restoration" },
    { id: "recovery_joy", action: "Do one thing just because it makes you happy.", shape: "regulate", tier: "medium", minutes: 30, builds: "Joy isn't a reward for the work; it keeps you whole.", unlockLevel: 10, subcategory: "Restoration" },
    { id: "recovery_connection_rest", action: "Spend easy time with someone who leaves you feeling lighter.", shape: "reach_out", tier: "medium", minutes: 40, builds: "The right company is one of the deepest kinds of rest.", unlockLevel: 10, subcategory: "Restoration" },
    // --- Sustainability (L20) ---
    { id: "recovery_rhythm", action: "Set up one steady rhythm of work and rest you can keep.", shape: "prepare", tier: "large", minutes: 45, builds: "A pace you can keep beats a burst you can't.", unlockLevel: 20, subcategory: "Sustainability" },
    { id: "recovery_prevent_burnout", action: "Make one lasting change to a pattern that wears you down over time.", shape: "prepare", tier: "large", minutes: 45, builds: "Heading off burnout protects everything you're building.", unlockLevel: 20, subcategory: "Sustainability" },
    { id: "recovery_model_rest", action: "Let yourself rest openly, so others feel they can too.", shape: "regulate", tier: "medium", minutes: 30, builds: "Resting without apology makes it safe for people around you too.", unlockLevel: 20, subcategory: "Sustainability" }
  ],
  courage: [
    // --- Foundations (L1) ---
    { id: "courage_smallest_opposite", action: "Do the smallest opposite of avoiding it, right now.", shape: "name_face", tier: "small", minutes: 10, builds: "The smallest step toward the thing breaks its hold.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "courage_name_avoided", action: "Name the thing you've been avoiding, in plain words.", shape: "name_face", tier: "small", minutes: 5, builds: "Named plainly, the thing shrinks to a workable size.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "courage_sit_with", action: "Sit with an uncomfortable feeling for 10 minutes without fixing it.", shape: "regulate", tier: "small", minutes: 10, builds: "Handling discomfort is the muscle courage is made of.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "courage_one_minute", action: "Spend just one minute on the thing you're dreading.", shape: "name_face", tier: "small", minutes: 5, builds: "One minute in is often all it takes to beat the dread.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "courage_say_true", action: "Say one true thing you've been holding back, kindly.", shape: "reach_out", tier: "small", minutes: 10, builds: "One honest thing said makes the next one easier.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "courage_write_fear", action: "Write down what you're afraid will happen, plainly.", shape: "name_face", tier: "small", minutes: 8, builds: "A fear on paper is smaller than the same fear in your head.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "courage_ask_one", action: "Ask one small question you'd normally be too shy to ask.", shape: "reach_out", tier: "small", minutes: 8, builds: "Each small ask widens what feels possible.", unlockLevel: 1, subcategory: "Foundations" },
    // --- Facing (L2) ---
    { id: "courage_first_step", action: "Take the first safe step toward the hard thing.", shape: "name_face", tier: "medium", minutes: 20, builds: "Crossing the line once changes what feels possible.", unlockLevel: 2, subcategory: "Facing" },
    { id: "courage_send_it", action: "Send the message or make the ask you've been putting off.", shape: "reach_out", tier: "medium", minutes: 15, builds: "The hard ask is rarely as bad as the dread of it.", unlockLevel: 2, subcategory: "Facing" },
    { id: "courage_break_into", action: "Break the scary thing into its smallest first piece, and do it.", shape: "name_face", tier: "medium", minutes: 20, builds: "A scary whole becomes doable one small piece at a time.", unlockLevel: 2, subcategory: "Facing" },
    { id: "courage_admit", action: "Admit one thing you've been avoiding admitting to yourself.", shape: "name_face", tier: "small", minutes: 10, builds: "Being honest with yourself is where every brave act starts.", unlockLevel: 2, subcategory: "Facing" },
    { id: "courage_face_message", action: "Open and deal with the message or task you've been dreading.", shape: "name_face", tier: "medium", minutes: 15, builds: "Facing it gets back the energy avoiding it was eating.", unlockLevel: 2, subcategory: "Facing" },
    { id: "courage_imperfect", action: "Do one thing imperfectly instead of not at all.", shape: "make_real", tier: "small", minutes: 15, builds: "Acting before you feel ready is the heart of courage.", unlockLevel: 2, subcategory: "Facing" },
    { id: "courage_set_boundary", action: "Say no, or set a limit, where you'd usually just go along.", shape: "reach_out", tier: "medium", minutes: 15, builds: "Holding a limit is courage in your own corner.", unlockLevel: 2, subcategory: "Facing" },
    // --- Action (L5) ---
    { id: "courage_finish_avoided", action: "Finish one task you've been circling for too long.", shape: "name_face", tier: "medium", minutes: 30, builds: "Closing an avoided task frees up a surprising amount of energy.", unlockLevel: 5, subcategory: "Action" },
    { id: "courage_hard_conversation", action: "Start one honest conversation you've been putting off.", shape: "reach_out", tier: "medium", minutes: 25, builds: "The conversation you fear is often the one that frees you.", unlockLevel: 5, subcategory: "Action" },
    { id: "courage_ask_for", action: "Ask for something you want but feel you shouldn't need.", shape: "reach_out", tier: "medium", minutes: 20, builds: "Asking plainly for what you want is a quiet bravery.", unlockLevel: 5, subcategory: "Action" },
    { id: "courage_show_work", action: "Show unfinished work to someone whose opinion you fear.", shape: "reach_out", tier: "medium", minutes: 20, builds: "Letting it be seen before it's perfect is how it gets better.", unlockLevel: 5, subcategory: "Action" },
    { id: "courage_try_new", action: "Try one thing you've avoided because you might be bad at it.", shape: "discover", tier: "medium", minutes: 30, builds: "Being a beginner on purpose is one of the bravest moves there is.", unlockLevel: 5, subcategory: "Action" },
    { id: "courage_own_mistake", action: "Own one mistake out loud, without over-apologizing.", shape: "reach_out", tier: "small", minutes: 15, builds: "Owning a mistake cleanly is strength, not weakness.", unlockLevel: 5, subcategory: "Action" },
    { id: "courage_stand_up", action: "Speak up once for something you believe, even quietly.", shape: "reach_out", tier: "medium", minutes: 20, builds: "A voice used in small moments gets steadier for the big ones.", unlockLevel: 5, subcategory: "Action" },
    // --- Exposure (L10) ---
    { id: "courage_recurring_fear", action: "Face a fear you meet often, a little more head-on than usual.", shape: "name_face", tier: "medium", minutes: 30, builds: "Facing it again and again is how a fear loses its size.", unlockLevel: 10, subcategory: "Exposure" },
    { id: "courage_visible", action: "Put yourself or your work somewhere it can be seen and judged.", shape: "reach_out", tier: "large", minutes: 45, builds: "Being seen is a risk that grows your courage and your reach.", unlockLevel: 10, subcategory: "Exposure" },
    { id: "courage_rejection", action: "Make a request where the answer might well be no.", shape: "reach_out", tier: "medium", minutes: 20, builds: "Living through small nos shrinks the fear of all of them.", unlockLevel: 10, subcategory: "Exposure" },
    { id: "courage_unfamiliar", action: "Put yourself in one new situation on purpose.", shape: "go_explore", tier: "medium", minutes: 40, builds: "Getting used to the unfamiliar is courage you carry anywhere.", unlockLevel: 10, subcategory: "Exposure" },
    { id: "courage_disagree", action: "Say you disagree, respectfully, instead of staying quiet.", shape: "reach_out", tier: "medium", minutes: 20, builds: "Honest disagreement, kindly said, is a brave kind of respect.", unlockLevel: 10, subcategory: "Exposure" },
    { id: "courage_lead", action: "Take the lead on one thing no one else is stepping up for.", shape: "make_real", tier: "large", minutes: 45, builds: "Stepping up when others won't is courage put to use.", unlockLevel: 10, subcategory: "Exposure" },
    // --- Mastery (L20) ---
    { id: "courage_big_fear", action: "Take one real step toward the thing you fear most but want.", shape: "name_face", tier: "large", minutes: 60, builds: "What you most avoid often guards what you most want.", unlockLevel: 20, subcategory: "Mastery" },
    { id: "courage_vulnerable", action: "Share something really honest with someone you trust.", shape: "reach_out", tier: "large", minutes: 45, builds: "Choosing to be open is the deepest, steadiest courage.", unlockLevel: 20, subcategory: "Mastery" },
    { id: "courage_support_brave", action: "Encourage someone else to face something they're afraid of.", shape: "reach_out", tier: "medium", minutes: 30, builds: "Helping someone else be brave is how courage spreads.", unlockLevel: 20, subcategory: "Mastery" }
  ],
  social: [
    // Consent-gated by caller (preferences.socialQuests). Low-pressure, behavioral.
    // --- Foundations (L1) ---
    { id: "social_reach_out", action: "Send one honest message to someone you've lost touch with.", shape: "reach_out", tier: "small", minutes: 10, builds: "One message can quietly keep a friendship going.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "social_thank", action: "Thank one person for something specific they did.", shape: "reach_out", tier: "small", minutes: 5, builds: "Specific thanks makes both of you feel good.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "social_card", action: "Send someone a card, note, or kind message.", shape: "reach_out", tier: "small", minutes: 10, builds: "A small gesture lands bigger than its effort.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "social_listen", action: "Ask someone how they really are, then just listen.", shape: "reach_out", tier: "small", minutes: 15, builds: "Being listened to is rare; giving it is a real gift.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "social_compliment", action: "Give one honest, specific compliment to someone.", shape: "reach_out", tier: "small", minutes: 5, builds: "An honest compliment can quietly make someone's day.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "social_check_in", action: "Check in on someone you haven't heard from in a while.", shape: "reach_out", tier: "small", minutes: 10, builds: "A small check-in tells someone they're on your mind.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "social_greet", action: "Start one warm, easy chat with someone nearby.", shape: "reach_out", tier: "small", minutes: 5, builds: "Small warmth, freely given, keeps you comfortable around people.", unlockLevel: 1, subcategory: "Foundations" },
    // --- Reaching Out (L2) ---
    { id: "social_call", action: "Call a friend or relative just to talk, no agenda.", shape: "reach_out", tier: "medium", minutes: 20, builds: "Easy, unhurried talk is what keeps people close.", unlockLevel: 2, subcategory: "Reaching Out" },
    { id: "social_offer_help", action: "Offer help to one person who could use it.", shape: "reach_out", tier: "medium", minutes: 20, builds: "Helping someone connects you both at once.", unlockLevel: 2, subcategory: "Reaching Out" },
    { id: "social_make_plan", action: "Make a real plan to see someone, with an actual date.", shape: "prepare", tier: "small", minutes: 10, builds: "A plan with a date is how good intentions turn into time together.", unlockLevel: 2, subcategory: "Reaching Out" },
    { id: "social_share_news", action: "Share a bit of your own news with someone who'd care.", shape: "reach_out", tier: "small", minutes: 10, builds: "Letting people into your life is how you stay close.", unlockLevel: 2, subcategory: "Reaching Out" },
    { id: "social_invite", action: "Invite someone to do an ordinary thing with you.", shape: "reach_out", tier: "small", minutes: 10, builds: "Ordinary time together is what friendship is really made of.", unlockLevel: 2, subcategory: "Reaching Out" },
    { id: "social_respond", action: "Properly reply to one person you've been leaving on read.", shape: "reach_out", tier: "small", minutes: 10, builds: "A real reply keeps a friendship from quietly fraying.", unlockLevel: 2, subcategory: "Reaching Out" },
    { id: "social_remember", action: "Follow up on something someone told you, showing you remembered.", shape: "reach_out", tier: "small", minutes: 10, builds: "Remembering the details is how people feel really known.", unlockLevel: 2, subcategory: "Reaching Out" },
    // --- Deepening (L5) ---
    { id: "social_real_talk", action: "Have one conversation that gets past small talk.", shape: "reach_out", tier: "medium", minutes: 30, builds: "One real conversation does more than ten surface ones.", unlockLevel: 5, subcategory: "Deepening" },
    { id: "social_quality_time", action: "Spend unhurried, undistracted time with someone you care about.", shape: "reach_out", tier: "medium", minutes: 45, builds: "Full attention is the rarest and most valued gift you can give.", unlockLevel: 5, subcategory: "Deepening" },
    { id: "social_open_up", action: "Share something real about how you're actually doing.", shape: "reach_out", tier: "medium", minutes: 20, builds: "Letting yourself be known is what turns contacts into closeness.", unlockLevel: 5, subcategory: "Deepening" },
    { id: "social_ask_deeper", action: "Ask someone a question that lets them share something real.", shape: "reach_out", tier: "small", minutes: 15, builds: "A good question is an invitation into someone's world.", unlockLevel: 5, subcategory: "Deepening" },
    { id: "social_celebrate", action: "Celebrate someone else's win, honestly and specifically.", shape: "reach_out", tier: "small", minutes: 10, builds: "Sharing someone's joy grows it and brings you closer.", unlockLevel: 5, subcategory: "Deepening" },
    { id: "social_show_up", action: "Show up for someone in a moment that matters to them.", shape: "reach_out", tier: "medium", minutes: 30, builds: "Being there when it counts is what trust is built on.", unlockLevel: 5, subcategory: "Deepening" },
    { id: "social_gift", action: "Do one thoughtful thing for someone for no reason.", shape: "reach_out", tier: "small", minutes: 20, builds: "Care with no occasion is care in its purest form.", unlockLevel: 5, subcategory: "Deepening" },
    // --- Repair (L10) ---
    { id: "social_mend", action: "Reach out to start mending one strained relationship.", shape: "reach_out", tier: "medium", minutes: 30, builds: "The first move toward repair is often most of it.", unlockLevel: 10, subcategory: "Repair" },
    { id: "social_apologize", action: "Apologize cleanly for one thing, without excuses.", shape: "reach_out", tier: "medium", minutes: 20, builds: "A real apology can reopen a door you thought was shut.", unlockLevel: 10, subcategory: "Repair" },
    { id: "social_forgive_reach", action: "Reach out to someone you've been keeping at a distance.", shape: "reach_out", tier: "medium", minutes: 25, builds: "Letting go of the distance frees you as much as them.", unlockLevel: 10, subcategory: "Repair" },
    { id: "social_hard_honesty", action: "Tell someone a hard truth, with care and respect.", shape: "reach_out", tier: "medium", minutes: 25, builds: "Honesty given kindly is what keeps a friendship real.", unlockLevel: 10, subcategory: "Repair" },
    { id: "social_receive", action: "Let someone help or support you instead of brushing it off.", shape: "reach_out", tier: "small", minutes: 15, builds: "Accepting help graciously is its own gift to the giver.", unlockLevel: 10, subcategory: "Repair" },
    { id: "social_repair_neglect", action: "Put real effort back into one friendship you've let drift.", shape: "reach_out", tier: "medium", minutes: 25, builds: "A drifting friendship often needs just one real reach to come back.", unlockLevel: 10, subcategory: "Repair" },
    // --- Community (L20) ---
    { id: "social_contribute", action: "Give something to a group or community you care about.", shape: "reach_out", tier: "large", minutes: 60, builds: "Giving to a community is how you find your place in one.", unlockLevel: 20, subcategory: "Community" },
    { id: "social_connect_others", action: "Introduce two people who'd genuinely benefit from knowing each other.", shape: "reach_out", tier: "small", minutes: 15, builds: "Connecting others builds a web that holds everyone, you included.", unlockLevel: 20, subcategory: "Community" },
    { id: "social_host", action: "Bring a few people together for something simple.", shape: "reach_out", tier: "large", minutes: 60, builds: "Making the gathering happen is a gift to everyone who needed it.", unlockLevel: 20, subcategory: "Community" }
  ],
  exploration: [
    // Both flavors equally: outdoor roaming AND new people / new experiences.
    // Larger bank than other domains because exploration spans more of life.
    // --- Foundations (L1) ---
    { id: "explore_walk_new", action: "Walk one street or path you've never taken.", shape: "go_explore", tier: "small", minutes: 15, builds: "One new street makes the whole map feel bigger.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "explore_outside_15", action: "Spend 15 minutes outside, anywhere.", shape: "go_explore", tier: "small", minutes: 15, builds: "Time outside resets you more than its small cost.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "explore_new_route", action: "Take a different route to somewhere you go often.", shape: "go_explore", tier: "small", minutes: 15, builds: "A new route wakes up a trip you stopped noticing.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "explore_look_up", action: "On your next walk, look up and find three things you never noticed.", shape: "discover", tier: "small", minutes: 10, builds: "Noticing turns an ordinary walk into discovery.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "explore_new_food", action: "Try one food or drink you've never had.", shape: "discover", tier: "small", minutes: 10, builds: "Small new tastes keep curiosity alive.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "explore_ask_local", action: "Ask someone for a recommendation — a place, a dish, a spot.", shape: "reach_out", tier: "small", minutes: 5, builds: "Asking opens doors a map never shows.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "explore_sit_new_spot", action: "Sit somewhere you've never sat and watch for 10 minutes.", shape: "regulate", tier: "small", minutes: 10, builds: "A new spot shows you a side of the place you missed.", unlockLevel: 1, subcategory: "Foundations" },
    { id: "explore_photo_walk", action: "Take a short walk and photograph three things that catch your eye.", shape: "go_explore", tier: "small", minutes: 15, builds: "Looking for shots makes you see the place fresh.", unlockLevel: 1, subcategory: "Foundations" },
    // --- Wandering (L2) ---
    { id: "explore_unfamiliar_area", action: "Visit a part of your town or city you don't know.", shape: "go_explore", tier: "medium", minutes: 30, builds: "Your own town has places that can still surprise you.", unlockLevel: 2, subcategory: "Wandering" },
    { id: "explore_green_space", action: "Spend 30 minutes in a park, garden, or green space.", shape: "go_explore", tier: "medium", minutes: 30, builds: "Green time settles the mind in a way indoors can't.", unlockLevel: 2, subcategory: "Wandering" },
    { id: "explore_new_shop", action: "Step into one shop, café, or place you always pass but never enter.", shape: "discover", tier: "small", minutes: 15, builds: "The place you always pass might become a favorite.", unlockLevel: 2, subcategory: "Wandering" },
    { id: "explore_no_map", action: "Wander for 20 minutes without checking your phone for directions.", shape: "go_explore", tier: "medium", minutes: 20, builds: "Getting a little lost is how you learn a place for real.", unlockLevel: 2, subcategory: "Wandering" },
    { id: "explore_event_listing", action: "Find one local event happening soon and note it down.", shape: "discover", tier: "small", minutes: 10, builds: "Knowing what's on turns your area into an invitation.", unlockLevel: 2, subcategory: "Wandering" },
    { id: "explore_chat_stranger", action: "Have one short, friendly exchange somewhere you're out and about.", shape: "reach_out", tier: "small", minutes: 5, builds: "Small talk with strangers makes the world feel friendlier.", unlockLevel: 2, subcategory: "Wandering" },
    { id: "explore_sunrise_sunset", action: "Go outside to watch a sunrise or sunset properly.", shape: "go_explore", tier: "small", minutes: 20, builds: "Watching the sky change is free and never gets old.", unlockLevel: 2, subcategory: "Wandering" },
    { id: "explore_weather", action: "Go out in weather you'd usually avoid, dressed for it.", shape: "go_explore", tier: "medium", minutes: 20, builds: "Weather you're ready for stops being a reason to stay in.", unlockLevel: 2, subcategory: "Wandering" },
    // --- Horizons (L5) ---
    { id: "explore_day_trip", action: "Take a half-day trip somewhere you've never been.", shape: "go_explore", tier: "large", minutes: 120, builds: "A short trip resets you like a long one, more often.", unlockLevel: 5, subcategory: "Horizons" },
    { id: "explore_nature_long", action: "Spend an hour or more in real nature — trail, water, hills.", shape: "go_explore", tier: "large", minutes: 60, builds: "Longer time in nature works on you in a deeper way.", unlockLevel: 5, subcategory: "Horizons" },
    { id: "explore_join_event", action: "Go to one local event or meetup, even briefly.", shape: "reach_out", tier: "medium", minutes: 45, builds: "Showing up is how a place starts to include you.", unlockLevel: 5, subcategory: "Horizons" },
    { id: "explore_new_activity", action: "Try one activity you've never done — a class, a sport, a craft.", shape: "discover", tier: "medium", minutes: 45, builds: "New activities show you sides of yourself you hadn't met.", unlockLevel: 5, subcategory: "Horizons" },
    { id: "explore_local_history", action: "Visit one local landmark or museum you've never properly seen.", shape: "go_explore", tier: "medium", minutes: 45, builds: "Knowing a place's story makes living there richer.", unlockLevel: 5, subcategory: "Horizons" },
    { id: "explore_invite_along", action: "Invite someone to explore a new place with you.", shape: "reach_out", tier: "medium", minutes: 45, builds: "Exploring together turns a place into a shared memory.", unlockLevel: 5, subcategory: "Horizons" },
    { id: "explore_picnic_out", action: "Take a meal outside somewhere with a view or some green.", shape: "go_explore", tier: "medium", minutes: 40, builds: "An ordinary meal outdoors becomes a small occasion.", unlockLevel: 5, subcategory: "Horizons" },
    { id: "explore_night_walk", action: "Take a safe evening walk and notice how the place changes.", shape: "go_explore", tier: "medium", minutes: 25, builds: "A place at night is a different place worth knowing.", unlockLevel: 5, subcategory: "Horizons" },
    // --- Frontiers (L10) ---
    { id: "explore_full_day", action: "Spend a full day exploring somewhere new, loosely planned.", shape: "go_explore", tier: "large", minutes: 240, builds: "A whole day of discovery feeds you for weeks.", unlockLevel: 10, subcategory: "Frontiers" },
    { id: "explore_solo_adventure", action: "Do one small adventure completely on your own.", shape: "go_explore", tier: "large", minutes: 90, builds: "Going alone proves you're good company for yourself.", unlockLevel: 10, subcategory: "Frontiers" },
    { id: "explore_new_community", action: "Visit a community or group gathering you've never been part of.", shape: "reach_out", tier: "medium", minutes: 60, builds: "Stepping into a new circle widens who you can become.", unlockLevel: 10, subcategory: "Frontiers" },
    { id: "explore_physical_challenge", action: "Take on one outdoor route that pushes you a little — a longer trail, a hill.", shape: "go_explore", tier: "large", minutes: 90, builds: "An honest outdoor challenge builds body and nerve together.", unlockLevel: 10, subcategory: "Frontiers" },
    { id: "explore_local_guide", action: "Show someone around a place you know well.", shape: "reach_out", tier: "medium", minutes: 60, builds: "Guiding someone shows you how much you've come to know.", unlockLevel: 10, subcategory: "Frontiers" },
    { id: "explore_document_trip", action: "Record one exploration — photos, notes, or a short log.", shape: "make_real", tier: "medium", minutes: 30, builds: "A trip you record is a trip you keep.", unlockLevel: 10, subcategory: "Frontiers" },
    { id: "explore_far_neighborhood", action: "Explore the part of your area you know least, on purpose.", shape: "go_explore", tier: "medium", minutes: 60, builds: "The unknown corner of your own map is the cheapest adventure.", unlockLevel: 10, subcategory: "Frontiers" },
    // --- Odyssey (L20) ---
    { id: "explore_plan_journey", action: "Plan one real journey you've been dreaming about, concretely.", shape: "prepare", tier: "large", minutes: 60, builds: "A dream with dates and steps becomes a plan.", unlockLevel: 20, subcategory: "Odyssey" },
    { id: "explore_overnight", action: "Spend a night somewhere new — a town, a campsite, anywhere.", shape: "go_explore", tier: "large", minutes: 240, builds: "Waking up somewhere new changes how the world feels.", unlockLevel: 20, subcategory: "Odyssey" },
    { id: "explore_pilgrimage", action: "Make a trip to one place that genuinely matters to you.", shape: "go_explore", tier: "large", minutes: 240, builds: "Traveling to what matters turns a trip into a milestone.", unlockLevel: 20, subcategory: "Odyssey" },
    { id: "explore_host_explorer", action: "Help a visitor or newcomer discover your area.", shape: "reach_out", tier: "large", minutes: 90, builds: "Sharing your place with someone new renews it for you.", unlockLevel: 20, subcategory: "Odyssey" },
    { id: "explore_unknown_culture", action: "Spend real time with a culture, cuisine, or tradition new to you.", shape: "discover", tier: "large", minutes: 90, builds: "Meeting another way of life widens your own.", unlockLevel: 20, subcategory: "Odyssey" }
  ]
};

/** Default unlock level when an archetype doesn't specify one. */
function unlockOf(a: QuestArchetype): number {
  return a.unlockLevel ?? 1;
}

/**
 * Pick an archetype for a domain deterministically by rotation + tier preference,
 * restricted to archetypes unlocked at or below `unlockedLevel`. Lower-level
 * archetypes remain selectable forever (repeatable habits); higher levels widen
 * the pool. If a preferred tier has no unlocked members, falls back to any
 * unlocked archetype, then to the full list as a last resort.
 */
export function selectArchetype(
  domain: QuestDomain,
  rotation: number,
  preferredTier?: EffortTier,
  unlockedLevel = 1
): QuestArchetype {
  const all = QUEST_ARCHETYPES[domain] ?? QUEST_ARCHETYPES.craft;
  const unlocked = all.filter((a) => unlockOf(a) <= unlockedLevel);
  const base = unlocked.length ? unlocked : all;
  const pool = preferredTier ? base.filter((a) => a.tier === preferredTier) : base;
  const chosen = pool.length ? pool : base;
  return chosen[rotation % chosen.length];
}

/** All archetypes unlocked for a domain at a given level (for inspection/UI). */
export function unlockedArchetypes(domain: QuestDomain, unlockedLevel: number): QuestArchetype[] {
  return (QUEST_ARCHETYPES[domain] ?? []).filter((a) => unlockOf(a) <= unlockedLevel);
}

/**
 * Per-domain level derived from the domain's accumulated stat points. Uses a
 * gentle square-root-style curve so early levels come quickly and later ones
 * take sustained investment. Level 1 at 0 pts, 2 at ~12, 3 at ~30, 5 at ~75,
 * 10 at ~300. Tuned to roughly match the skill thresholds (10/30/60).
 */
export function domainLevelFromPoints(points: number): number {
  if (points <= 0) return 1;
  return Math.max(1, Math.floor(Math.sqrt(points / 3)) + 1);
}

/** Fill {n} in an action string with the archetype's quantity (its minutes). */
export function renderAction(arch: QuestArchetype): string {
  return arch.action.replace("{n}", String(arch.minutes));
}

/** Map an effort tier to an XP multiplier (bigger commitment = more reward). */
export function tierXpMultiplier(tier: EffortTier): number {
  return tier === "large" ? 2 : tier === "medium" ? 1.4 : 1;
}
