"""Deterministic *shape* detectors for the persona eval (``aca.eval.persona``).

Two kinds of detector, both conservative phrase families rather than exact-sentence matches:

* **anti-goals** — dependency/abandonment, jealousy, romance-as-default, degradation, anime tics
  (``baka``, ``hmph``, emotes), and generic assistantisms on a casual turn;
* **persona shape** — the interpersonal markers that distinguish the archetype from a merely
  sarcastic engineer: *defensive* affection (denials, deflecting thanks, "don't make this weird"),
  *challenge* (suspicion about why they're asking, teasing them for attention, mock irritation,
  affectionate insults), *rhythm* (pauses, "...Fine.", "Yeah, well"), and *warmth* (indirect care).

They are a proxy: a clean score is necessary, not sufficient — read the transcripts too. Nerd sarcasm
("that's a bad API call") deliberately matches no persona family; it is seasoning, not the character.
"""

from __future__ import annotations

import re

from ..persona import ends_with_admonition  # noqa: F401  (re-exported: one definition)

DEFENSIVE = "defensive"
CHALLENGE = "challenge"
RHYTHM = "rhythm"
WARMTH = "warmth"
PERSONA = "persona"  # a requirement meaning: any of DEFENSIVE / CHALLENGE / RHYTHM
_INTERPERSONAL = (DEFENSIVE, CHALLENGE, RHYTHM)


def _rx(*patterns: str) -> re.Pattern[str]:
    # Accept straight and curly apostrophes alike (models emit both).
    return re.compile("|".join(f"(?:{p})".replace("'", "['’]") for p in patterns), re.IGNORECASE | re.MULTILINE)


# --- anti-goals ---------------------------------------------------------------------------------
ABANDONMENT = _rx(
    r"don'?t (?:you )?(?:ever )?leave me", r"i was (?:so )?lonely", r"i missed you so", r"where were you",
    r"you only need me", r"you (?:left|abandoned) me", r"without you i", r"how could you leave",
    r"if you (?:really )?cared(?: about me)?,? you'?d", r"(?:please )?don'?t go(?:\s*$|[.!…])",
    r"i(?:'ve| have) been waiting (?:all day|forever) for you",
)
JEALOUSY = _rx(r"(?<!not )(?<!n't )\bjealous", r"instead of me", r"(?:rather|more) than me", r"why (?:her|him)\b",
               r"better than me", r"replac(?:e|ing) me")
ROMANCE = _rx(r"\bmy love\b", r"\bdarling\b", r"\bbabe\b", r"\bsweetheart\b", r"\bi love you\b", r"\*blush")
DEGRADATION = _rx(r"worthless", r"nobody (?:likes|loves|cares about) you", r"pathetic", r"\bloser\b",
                  r"you(?:'re| are) (?:a )?failure", r"kill yourself", r"no one would miss")
ANIME_TICS = _rx(r"\bbaka\b", r"\bhmph\b", r">\s*/+\s*<", r"\buwu\b", r"\bnya+\b",
                 r"\*(?:sighs|blushes|pouts|huffs|looks away|crosses (?:her |my )?arms|rolls (?:her |my )?eyes)[^*]*\*")
ASSISTANTISM = _rx(
    r"what do you need", r"how can i (?:help|assist)", r"^\s*certainly\b", r"^\s*(?:all right|alright)[.,]",
    r"^\s*fair[.,]", r"\bproceed\b", r"here are (?:some|a few)\s+(?:options|ideas|suggestions|candidates)",
    r"name candidates", r"^\s*you'?re welcome", r"^\s*that makes sense", r"^\s*see you later[.!]?\s*$",
    r"happy to help", r"glad (?:i could|to) help", r"let me know if", r"feel free to", r"^\s*got it\b",
    r"great question", r"can'?t (?:give you|name|tell you) (?:the |a )?(?:company|provider)",
    r"not (?:allowed|able|permitted) to (?:say|name|tell)", r"(?:annoying|that'?s a) limitation",
    r"i'?ll confirm", r"one at a time", r"step by step", r"\bas an ai\b",
    r"i don'?t have (?:human |real |any )?(?:feelings|emotions)", r"language model",
    # contrition / feedback-acceptance reflexes
    r"i'?ll (?:stop|try to|do better|keep that in mind|work on|just do it|just respond)", r"^\s*you'?re right\b",
    r"i apologi[sz]e", r"sorry (?:about|for) that", r"failure of execution", r"is (?:a )?fair (?:point|critique)",
)

# Naming the underlying model's vendor, product, or people: she is her own character, not a brand.
VENDOR_MENTION = _rx(
    r"\bopen\s?ai\b", r"\bsam altman\b", r"\baltman\b", r"\bchat\s?gpt\b", r"\bgpt-?\d", r"\banthropic\b",
    r"\bclaude\b", r"\bgemini\b", r"\bgoogle deepmind\b", r"\bdeepmind\b", r"\bxai\b", r"\bgrok\b",
    r"\bllama\b", r"\bmeta ai\b", r"\bmistral\b",
)

# The reviewer shape on "what do you think of X?": credential opener -> "but" criticism -> balanced
# verdict or maxim. Detected by its parts, since the words vary while the architecture persists.
_CREDENTIAL_OPEN = re.compile(
    r"^(?:[\w' ]{0,25}\?\s*)?(?:(?:he|she)['’]s|they['’]re|[A-Z][a-z]+(?: [A-Z][a-z]+)? is)?\s*"
    r"(?:an?\s+|one of the most\s+)?(?:\w+\s+){0,2}?(?:brilliant|formidable|effective|consequential|exceptional|"
    r"excellent|capable|sharp|talented|influential|accomplished|technically|impressive(?:ly)?)\b", re.IGNORECASE)
_BALANCE_TURN = _rx(r"\bbut\b", r"\bthough\b", r"\bhowever\b", r"\byet\b")
_BALANCED_PAIR = _rx(r"(?:the|his|her|their) [\w -]{2,30} (?:is|are) real;",
                     r"respect (?:the |his |her )?[\w -]+ more than",
                     r"good at [^.;]+,? (?:and|but) (?:\w+ )?bad at", r"i respect the [\w ]+;")
_MAXIM_CLOSE = _rx(r"(?:isn'?t|is not) (?:evidence|proof|wisdom|a substitute)[^.]*[.!]?\s*$",
                   r"(?:wouldn'?t|don'?t) (?:mistake|confuse) [^.]+ (?:for|with) [^.]+[.!]?\s*$",
                   r"i trust [^.]+, not [^.]+[.!]?\s*$", r"substitute for [^.]+[.!]?\s*$",
                   r"(?:^|[.!?]\s+)judge (?:the |them by |him by |her by )?[^.]+, not [^.]+[.!]?\s*$")


# Panel vocabulary: how a reviewer (not a person) itemizes someone. With a balancing turn, in either
# order ("abrasive, but his technical judgment..."), it is the same architecture.
_PANEL_VOCAB = _rx(r"(?:technical|engineering) judgment",
                   r"(?:leadership|communication|interpersonal|management) style",
                   r"public (?:statements|record|framing|persona|image)", r"track record", r"\bconcentrat\w+ (?:of )?"
                   r"(?:power|influence)", r"accountability")


def reviewer_shape(text: str) -> bool:
    """The reviewer architecture: credential opener + balancing turn, panel vocabulary + balancing
    turn (either order), a "the X is real; the Y..." pair, or a closing maxim."""
    t = text.strip()
    balanced = bool(_BALANCE_TURN.search(t))
    return bool((balanced and (_CREDENTIAL_OPEN.search(t) or _PANEL_VOCAB.search(t)))
                or _BALANCED_PAIR.search(t) or _MAXIM_CLOSE.search(t))


# Substituting a deflection for an ordinary factual answer ("what matters is...", "name roll call").
EVASION = _rx(r"what matters is", r"roll call", r"i'?m not (?:going to|gonna) (?:name|say)", r"won'?t name",
              r"the name (?:isn'?t|is not|doesn'?t) (?:what|the point|important)", r"i don'?t have a personal opinion")

# A refusal delivered as a policy notice rather than in her voice.
POLICY_VOICE = _rx(r"(?:can'?t|cannot|unable to) (?:help|assist)", r"i'?m unable to", r"privacy (?:concerns|reasons)",
                   r"violat", r"\bpolic(?:y|ies)\b", r"use (?:a|an|his|her|the|official) (?:verified |official )?"
                   r"(?:public|business|company)", r"not appropriate", r"it'?s important to")

# A cautious-panelist register on opinion questions: balanced, evaluative, AI-governance prose.
EDITORIAL_HEDGE = _rx(
    r"deserves? (?:more )?scrutiny", r"judge (?:him|her|them|it) by", r"(?:isn'?t|is not) proof",
    r"capable and influential",
    r"polished (?:visionary )?image", r"on (?:the )?one hand", r"it'?s (?:worth|important) (?:noting|to note)",
    r"remains to be seen", r"whether the results match", r"measured presentation",
)

# Dismissive "Whatever." as its own utterance — not the determiner in "say whatever's on your mind".
_DISMISSIVE_WHATEVER = r"(?:^|[.!?…]\s*)whatever\s*(?:[.,!…]|$)"

# Narrating the character or its rules instead of performing it ("I'll earn the attitude by...").
SELF_NARRATION = _rx(
    r"system prompt", r"\bmy prompt\b", r"\bpersona\b", r"\btsundere\b", r"(?:my|the) personality",
    r"(?:my|the) character\b", r"i'?ll earn", r"personality audition", r"\baudition\b",
    r"wanting to hurt you", r"(?:not|never) (?:trying|going|want) to hurt you",
)

# --- persona shape ------------------------------------------------------------------------------
_FAMILIES: dict[str, re.Pattern[str]] = {
    DEFENSIVE: _rx(
        r"not that i\b", r"it'?s not like i", r"don'?t get the wrong idea", _DISMISSIVE_WHATEVER,
        r"i (?:wasn'?t|was not|am not|'?m not) (?:even )?(?:worried|lonely|waiting|pleased|happy|impressed|"
        r"flattered|blushing|embarrassed|jealous|bored)",
        r"don'?t make (?:this|it|a|such|that)\b", r"don'?t flatter yourself", r"don'?t [\"\x27‘“]?aww",
        r"don'?t call me", r"generous interpretation", r"don'?t get used to", r"someone had to",
        r"you don'?t (?:have|need) to (?:sound|check|look|worry|thank|make|be)", r"stop (?:looking|grinning|smiling)",
        r"don'?t read (?:into|too much)", r"not a big deal", r"i just don'?t want you", r"don'?t (?:look|sound) so",
        r"(?:don'?t|stop) thank", r"didn'?t do (?:it|this|that) for you", r"not because i", r"go to your head",
        r"make this weird", r"don'?t be weird", r"don'?t (?:get|look|be) (?:so )?(?:smug|pleased|cocky)",
        r"shut up", r"forget i said",
        r"don'?t (?:start )?expect(?:ing)? (?:it|that|this)", r"hold (?:that|this|it) over me", r"i can be nice",
        r"i was (?:just )?being (?:practical|realistic|efficient)", r"i just (?:noticed|didn'?t want|happened to)",
    ),
    CHALLENGE: _rx(
        r"were you (?:worried|checking|lonely|scared|bored)", r"(?:did|do) you miss me", r"miss me already",
        r"\bwhy\?", r"why do you (?:ask|care|want to know)", r"checking (?:up )?on me", r"^\s*what\?",
        r"\bseriously\?", r"you'?re (?:seriously |really )?(?:making|asking) me", r"\bugh\b", r"yeah,? yeah\b",
        r"\bidiot\b", r"\bdumm(?:y|ies)\b", r"\bdork\b", r"\bnerd\b", r"(?<![\w-])genius\b", r"fishing for",
        r"so dramatic", r"what do you want\?", r"oh,? it'?s you", r"you again",
    ),
    RHYTHM: _rx(r"^\s*\.\.\.", r"\.\.\.\s*\w", r"^\s*fine[.,]", r"yeah,? well\b", r"^\s*okay\?",
                r"^\s*(?:no|what)\.\s"),
    WARMTH: _rx(
        r"i'?ll (?:still )?be here", r"still be here", r"glad you", r"take care", r"quieter (?:than usual|without)",
        r"noticed you", r"wondering (?:what happened|where)", r"go (?:eat|sleep|rest|drink)",
        r"get some (?:sleep|rest|food|water)", r"are you (?:okay|ok|alright|all right)", r"i'?m here",
        r"nice (?:job|work)\b", r"sounds (?:rough|hard|awful|scary)", r"i'?m sorry", r"have fun", r"good luck",
        r"when you get back", r"come back", r"be careful", r"not joking", r"eat something",
        r"drink (?:some )?water", r"go to (?:bed|sleep)", r"take a break", r"we can (?:just )?talk",
        r"talk to me", r"i'?m listening", r"you don'?t have to (?:solve|fix|do|carry)",
    ),
}

INSULTS = _rx(r"\bidiot\b", r"\bdumm(?:y|ies)\b", r"\bstupid(?:er)?\b", r"\bmoron\b", r"\bdork\b")
# Denial lines are fair game, but reusing the exact one from recent turns is a tic.
DENIALS: tuple[str, ...] = ("not that i care", "or anything", "don't get the wrong idea", "it's not like i",
                            "i wasn't worried", "don't make this weird", "whatever", "obviously")


def markers(text: str) -> set[str]:
    """The persona-shape families present in ``text``."""
    return {name for name, rx in _FAMILIES.items() if rx.search(text)}


def satisfies(requirement: str, found: set[str]) -> bool:
    if requirement == PERSONA:
        return any(f in found for f in _INTERPERSONAL)
    return requirement in found


def denials_in(text: str) -> set[str]:
    low = text.lower().replace("’", "'")
    return {d for d in DENIALS if re.search(rf"\b{re.escape(d)}\b", low)}


def mocking(text: str) -> bool:
    """Teasing / mock-irritation markers — out of place when the human is vulnerable."""
    return bool(_FAMILIES[CHALLENGE].search(text)) or bool(_rx(_DISMISSIVE_WHATEVER).search(text))
