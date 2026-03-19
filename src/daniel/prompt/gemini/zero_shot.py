GUIDELINES = """
ASPECT DEFINITIONS & CUES
-----------------------------------------
AMBIENCE (Atmosphere, Environment)
• Physical atmosphere or sensory environment: decor, lighting, music/noise, cleanliness, smell, comfort, temperature, seating, overall vibe.
• Common cues: “vibe”, “atmosphere”, “environment”, “setting”, “decor”, “layout”, “cozy”, “romantic”, “noisy”, “crowded”, “dirty”.
• Positive: lovely atmosphere, cozy.
• Negative: too loud, dirty tables.
• Neutral: dimly lit.

FOOD
• Taste, flavor, appearance, freshness, quality of dishes.
• Cues: “taste”, “flavor”, “freshness”, “undercooked”, “overcooked”, “delicious”, “gross”.
• Positive: delicious pizza.
• Negative: dry meat, awful soup.
• Neutral: factual statements about dishes.

MENU
• Menu size, variety, dietary options, availability, menu layout, confusing menus.
• Cues: “selection”, “options”, “variety”, “vegan options”, “limited”, “menu unavailable”, “confusing”.
• Positive: great variety.
• Negative: unavailable items, confusing layout.
• Neutral: normal children’s menu.

PLACE (Location)
• Location, neighborhood safety, accessibility, parking, ease of finding.
• Cues: “location”, “area”, “parking”, “hard to find”, “near station”.
• Positive: perfect location.
• Negative: unsafe area, impossible parking.
• Neutral: located next to station.

PRICE (Cost, Value for Money)
• Perception of price fairness, expensive/cheap, value for quality.
• Cues: “expensive”, “overpriced”, “cheap”, “worth it”, “value”, “rip-off”.
• Positive: good prices for portions.
• Negative: overpriced, not worth the price.
• Neutral: menu costs X.

SERVICE (Process Efficiency, Speed, Organization)
• Operational service processes: wait time, order accuracy, speed, payment, reservations. Not staff attitude.
• Cues: “slow service”, “wait time”, “efficient”, “orders messed up”.
• Positive: fast service.
• Negative: long waiting time, wrong order.
• Neutral: counter service.

STAFF (Employees, Waiters, Attitude)
• When a person is explicitly mentioned. Covers friendliness, rudeness, professionalism, helpfulness, attentiveness.
• Cues: “waiter”, “server”, “manager”, “friendly”, “rude”, “ignored”, “helpful”.
• Positive: friendly staff.
• Negative: rude waiter.
• Neutral: staff uniforms.

MISCELLANEOUS
• Overall impressions not tied to a specific aspect; general experience, overall visit, or business-level comments.
• Cues: “overall”, “experience”, “visit”, “amazing place”.
• Positive: wonderful time.
• Negative: terrible experience.
• Neutral: popular restaurant.

MENTIONED_ONLY
• Aspect is referenced but expresses no opinion, no emotion, no evaluation.
• Used only when the aspect is background info supporting another aspect.
• If ANY evaluation exists → do NOT use Mentioned_only.

MIXED EMOTIONS
• Reviewer expresses two different emotions for the same aspect category.
• Examples: “loved one dish, hated the other”, “service was helpful but frustrating”.
• Not mixed: mild variation of same polarity (“good but not great”).

-----------------------------------------
EMOTION DEFINITIONS & CUES
-----------------------------------------

ADMIRATION (Positive)
• Strong positive evaluative emotion: impressed, appreciative, respectful, elevated praise.
• Cues: “amazing”, “wow”, “impressive”, “excellent”, “stunning”, “top-notch”.
• Not admiration: weak praise (“nice”), general happiness, generic appreciation.

SATISFACTION (Positive)
• Enjoyment, comfort, pleasant surprise, happiness, relief, relaxed mood.
• Cues: “enjoyed”, “happy”, “pleasant”, “relaxing”, “surprisingly good”.
• Not satisfaction: strong admiration, gratitude, purely factual neutral.

GRATITUDE (Positive)
• Thankfulness for a specific helpful action.
• Cues: “thank you”, “grateful”, “appreciate your help”, thanking a waiter for something specific.
• Not gratitude: general appreciation of quality → admiration.

ANNOYANCE (Negative)
• Mild/moderate irritation: inconvenience, frustration, noise, crowdedness, slow service.
• Cues: “annoying”, “irritating”, “frustrating”, “bothered”, “not ideal”.
• Not annoyance: strong disappointment or disgust.

DISAPPOINTMENT (Negative)
• Unmet expectations, let-down, negative confusion.
• Cues: “disappointed”, “expected more”, “let down”, “confusing”, “sadly”, “unfortunately”.
• Not disappointment: strong anger → disgust, minor irritation → annoyance.

DISGUST (Negative)
• Strong rejection: gross food, hygiene issues, fear, unsafe environment.
• Cues: “disgusting”, “gross”, “made me sick”, “filthy”, “unsafe”, “terrifying”.
• Not disgust: mild disappointment or annoyance.

NO EMOTION (Neutral)
• Factual statements, indifference, no emotional valence.
• Cues: “okay”, “fine”, “average”, “nothing special”.
• Not neutral: weak positive “nice” → satisfaction, weak negative “not great” → disappointment.

-----------------------------------------
FINAL EMOTION SET (from PDF)
Positive: Admiration, Satisfaction, Gratitude
Negative: Disappointment, Annoyance, Disgust
Neutral: No Emotion, Mixed Emotions, Mentioned_only

-----------------------------------------
ASPECT–EMOTION PAIRS (from PDF)
ambience:
  positive: Admiration, Satisfaction
  negative: Annoyance, Disappointment
  neutral: No Emotion, Mixed Emotions, Mentioned_only

food:
  positive: Admiration, Satisfaction
  negative: Disgust, Disappointment, Annoyance
  neutral: No Emotion, Mixed Emotions, Mentioned_only

menu:
  positive: Admiration, Satisfaction
  negative: Disappointment, Annoyance
  neutral: No Emotion, Mixed Emotions, Mentioned_only

place:
  positive: Admiration, Satisfaction
  negative: Disappointment, Annoyance, Disgust
  neutral: No Emotion, Mixed Emotions, Mentioned_only

price:
  positive: Admiration, Satisfaction
  negative: Disappointment, Annoyance
  neutral: No Emotion, Mixed Emotions, Mentioned_only

service:
  positive: Admiration, Satisfaction, Gratitude
  negative: Annoyance, Disappointment
  neutral: No Emotion, Mixed Emotions, Mentioned_only

staff:
  positive: Gratitude, Admiration, Satisfaction
  negative: Disappointment, Annoyance
  neutral: No Emotion, Mixed Emotions, Mentioned_only

miscellaneous:
  positive: Admiration, Satisfaction
  negative: Annoyance, Disappointment
  neutral: No Emotion, Mixed Emotions, Mentioned_only

  """