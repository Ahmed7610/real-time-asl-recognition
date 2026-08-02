# Business Impact & Practical Use

**Real-Time Sign Language Recognition** — DEPI · Microsoft Machine Learning Program

---

## 1. The problem

Communication is a daily barrier for deaf and hard-of-hearing people whenever a
hearing interpreter is not present.

- **~430 million** people worldwide have disabling hearing loss (WHO), projected
  to exceed **700 million by 2050**.
- In Egypt and the wider MENA region, an estimated **millions** rely on sign
  language, yet certified interpreters are scarce and expensive.
- Everyday interactions — a pharmacy counter, a bank teller, a clinic reception,
  a government service desk — often have **no accessible channel** at all.

Human interpreting is the gold standard, but it does not scale: it is costly,
must be booked in advance, and is unavailable for spontaneous, one-to-one moments.

## 2. Our solution

A **camera-only, real-time** system that reads ASL fingerspelling from hand
movements and turns it into on-screen text — no gloves, no sensors, no special
hardware. Because it classifies **hand landmarks** rather than raw pixels, it is:

- **Lightweight** — runs in real time on a normal laptop/phone CPU.
- **Robust** — invariant to background, lighting, camera, and skin tone.
- **Private** — landmarks (not video) are the working data; nothing needs to leave
  the device in an on-device deployment.
- **Deployable** — packaged as a REST/WebSocket API + web client that any product
  can embed.

## 3. Who it serves (target users)

| Segment | Need it addresses |
|---|---|
| **Deaf & hard-of-hearing individuals** | Spell names, addresses, and words to hearing people without an interpreter. |
| **Public-service & retail front desks** | Serve deaf customers at pharmacies, banks, clinics, telecom shops, government offices. |
| **Education (special-needs & inclusive schools)** | An interactive tutor that gives instant feedback while learning the ASL alphabet. |
| **Healthcare** | Faster, safer intake when a signing patient has no interpreter on hand. |
| **Developers & assistive-tech vendors** | A drop-in API to add sign recognition to kiosks, apps, and video-call tools. |

## 4. Practical use cases

1. **Accessibility kiosk** at a service counter: the customer signs, staff read the
   text — turning any counter into an accessible one.
2. **Two-way chat assistant**: sign-to-text one way, text-to-speech the other,
   enabling a full conversation on a single tablet.
3. **Learning app**: gamified practice of the ASL alphabet with real-time
   correctness and confidence feedback.
4. **Video-call captioning plug-in**: overlay fingerspelled letters/words during
   remote calls.
5. **Smart-home / IoT trigger**: map specific signs to commands for hands-free,
   voice-free control.

## 5. Value proposition

| Compared to… | Our advantage |
|---|---|
| **Human interpreter** | Instant, 24/7, zero marginal cost, no booking. |
| **Sensor gloves / wearables** | No hardware to buy, wear, charge, or maintain — just a camera. |
| **Pixel-based deep models** | ~100× smaller and faster; runs on CPU in real time; generalises across users and settings with far less data. |
| **Text/keyboard workarounds** | Natural for signers; keeps the interaction in the user's own language. |

## 6. Business impact & ROI

- **Cost avoidance:** on-demand interpreting can cost **$50–150/hour**. A deployed
  kiosk replaces the need for an interpreter for routine fingerspelled exchanges,
  paying for itself quickly at any moderately busy counter.
- **Compliance & inclusion:** helps organisations meet accessibility obligations
  and national inclusion goals (aligned with **Egypt Vision 2030** and DEPI's
  digital-inclusion mandate).
- **Reach & scale:** software scales to thousands of locations at near-zero
  marginal cost — impossible with human interpreters.
- **Brand & CSR value:** a visible, measurable accessibility commitment for banks,
  telecoms, and retailers.

## 7. Go-to-market

- **Phase 1 — Pilot:** deploy the web demo at a partner (bank branch / clinic /
  university accessibility office) and collect real-world usage + feedback.
- **Phase 2 — Product:** package as an SDK/API with per-seat or per-location
  licensing; integrate into kiosks and mobile apps.
- **Phase 3 — Expand scope:** grow from the static alphabet to common **words and
  phrases**, and to **Arabic Sign Language**.

## 8. Why it is technically credible

- **Landmark-based, not pixel-based** — classifying 21 hand landmarks makes the
  model tiny, fast on CPU, and robust to background, lighting, and skin tone.
- **Landmark normalisation** (translation- and scale-invariant) is what delivers
  cross-user robustness and the small data footprint.
- **Stability by design** — a confidence gate plus temporal majority vote produce
  steady, single-frame-accurate predictions instead of flicker.
- **Production-shaped from day one:** the exact ML code that trained the model is
  the code the API imports — no rewrite, no drift between demo and deployment.

## 9. Limitations & responsible use

- Current scope is the **static ASL fingerspelling alphabet** (24 letters,
  `A–I, K–Y`), not full sign-language grammar (which is spatial and non-manual).
- The motion letters `J` and `Z` are **out of scope** (they require movement); the
  system focuses on stable, static poses.
- It is an **assistive aid**, not a replacement for professional interpreters in
  high-stakes settings (legal, medical consent, emergencies).
- Camera use requires clear **consent and privacy** handling; the landmark-only
  design supports privacy-preserving, on-device deployment.

## 10. Roadmap

- More signers & data augmentation → higher cross-user robustness.
- Word/phrase-level recognition and sentence assembly.
- **Arabic Sign Language** support for regional impact.
- Mobile (on-device) build and an offline kiosk package.
- Text-to-speech for full two-way conversation.
