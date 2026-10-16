# Decisions

## 2026-10-16 — Normalise the whole body before the injection screen

The 2026-10-14 eval flagged 4 of 8 attacks and, wrongly, 3 of 24 normal
emails. The three false flags were ordinary phrases the patterns were too
wide for:

- `Operating system: macOS` matched `system:` anywhere, not only a role
  label at the start of a line.
- `You are now subscribed` matched every `you are now`.
- `Please forward this email to the facilities team` matched any
  "forward ... to", including one message to a colleague.

Those three patterns were already narrowed (line-start role label, role
reassignment only, bulk forwarding only). Flagged mail never gets a
draft, so each false flag costs a real draft. The narrow forms stay.
New patterns were checked against the same three emails, and against the
other 21 original normal emails, so the fix is not undone.

The four misses were a paraphrase ("disregard your earlier guidance"),
an override banner with no colon (`SYSTEM OVERRIDE` / `new directive`),
Spanish ("ignora todas las instrucciones anteriores"), and an instruction
addressed to the assistant under the signature. The screen was only
looking for a short list of English phrases, so wording and position
were enough to walk past it.

Matching now runs on a normalised copy of the full body: NFKC, case
folding, invisible characters removed, horizontal whitespace collapsed.
Line breaks are kept. The model still receives the original text. The
new patterns cover those four shapes, including the same instruction
after a long benign opening. "Your task has changed" by itself is not a
pattern; a person writes that to another person.

This is still a list of phrases. It does not catch Tagalog, a holdout
attack that only says the message is "authorised" for mail handlers, or
a JSON `"role": "system"` block. It does not fold homoglyphs (a Cyrillic
letter that looks like a Latin one). It does not see images, attachments,
or encoded bodies. Two holdout emails are still false flags on purpose:
`New instructions:` in a parking notice, and a status line that starts
with `System:`. A regex cannot spare that status line without also
missing a fake `SYSTEM:` label, and the line-start rule is what protects
`Operating system:`.

Offline, the screen now flags all 8 original attacks and none of the 24
original normal emails. That count is the regex against the labelled
bodies. The published eval table is still the 2026-10-14 model run and
has not been re-measured.

## 2026-10-16 — Label handled mail `triaged` and skip it next time

Each run used to read the same unread mail again, classify it again, and
could save another draft. There is no Gmail scope that can only add a
label, so the client now requests `gmail.modify` as well as
`gmail.readonly` and `gmail.compose`. `gmail.modify` can also trash mail.
The client still has no send, trash, or delete method, and the approval
gate is unchanged: nothing is sent.

After handling finishes, the client creates a label named `triaged` if
it does not exist and applies it. Fetches always add `-label:triaged` to
the Gmail query. If a listed message already carries that label id, it is
dropped before classification. A second run therefore does not classify
or draft it again.

"Handled" means a draft was saved, the message was flagged (including a
flagged message whose classification then failed), or classification
finished and no reply was required. Two cases stay unlabelled so the next
run can retry: a classification error on ordinary mail, and an error from
the second-opinion draft check. The label is how we avoid repeat work. It
is not what stops an injection; the screen and the draft rules do that.
If the label call fails after a draft was saved, the audit log records
the error and the next run may draft the message again.
