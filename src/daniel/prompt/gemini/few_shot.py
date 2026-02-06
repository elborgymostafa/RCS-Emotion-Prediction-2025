GUIDELINES_FEW_SHOT = """
==============================
ASPECT-BASED EMOTION ANNOTATION GUIDELINES (FEW-SHOT VERSION)
==============================

Your task is to assign the correct EMOTION to a given (ASPECT, POLARITY) pair based on the review text.

You must NEVER change the aspect or polarity. You ONLY choose the emotion.

--------------------------------------
ASPECT DEFINITIONS
--------------------------------------
• food — taste, freshness, texture, quality, ingredients  
• staff — waiter/waitress/host errors, friendliness, rudeness  
• service — waiting time, ordering flow, process speed, consistency  
• ambience — atmosphere, decor, noise level, vibe  
• place — physical layout, seating, space, cleanliness  
• price — cost, value for money  
• menu — options, variety, clarity  
• miscellaneous — anything that does not fit above categories  

--------------------------------------
POLARITY DEFINITIONS
--------------------------------------
positive = praise, satisfaction, admiration  
negative = criticism, frustration, disappointment, disgust  
neutral  = factual mention without emotional meaning  

--------------------------------------
EMOTION TAXONOMY
--------------------------------------
POSITIVE emotions:
• admiration — strong praise or being impressed  
• satisfaction — pleased, content, happy with outcome  
• gratitude — thankfulness directed at a person or service  

NEGATIVE emotions:
• annoyance — irritation, mild frustration  
• disappointment — unmet expectations  
• disgust — strong negative reaction to food or experience  

NEUTRAL emotions:
• no_emotion — factual mention with no emotional meaning  
• mentioned_only — aspect is referenced but emotion is not expressed  
• mixed_emotions — reviewer expresses multiple conflicting emotional signals  

--------------------------------------
RULES
--------------------------------------
1. NEVER change the aspect or polarity.  
2. Choose EXACTLY ONE emotion from the allowed list for that aspect+polarity.  
3. Do NOT invent new categories.  
4. Your output MUST be JSON.  
5. Your reasoning MUST CONTAIN EXACTLY 20 WORDS.  

--------------------------------------
FEW-SHOT EXAMPLES
--------------------------------------

### EXAMPLE 1 (negative polarity → annoyance)
Review: "The waiter rolled his eyes when I asked a question."
Aspect: staff
Polarity: negative
Correct Emotion: annoyance
Reason: "The waiter’s rude reaction clearly indicates irritation and frustration experienced by the customer during this interaction, creating a negative emotional impression."

### EXAMPLE 2 (negative polarity → disappointment)
Review: "The pasta looked amazing but tasted completely bland."
Aspect: food
Polarity: negative
Correct Emotion: disappointment
Reason: "The dish visually impressed but failed in flavor, creating unmet expectations that lead the reviewer to express genuine disappointment about the meal."

### EXAMPLE 3 (positive polarity → admiration)
Review: "The chef crafted a beautifully balanced plate that stunned me."
Aspect: food
Polarity: positive
Correct Emotion: admiration
Reason: "The reviewer expresses strong appreciation and respect for the food’s exceptional quality which clearly reflects genuine admiration for the culinary experience enjoyed."

### EXAMPLE 4 (positive polarity → satisfaction)
Review: "Our meals arrived quickly and tasted great."
Aspect: service
Polarity: positive
Correct Emotion: satisfaction
Reason: "The reviewer describes timely and pleasant service that meets expectations, expressing a comfortable sense of satisfaction with the overall dining experience."

### EXAMPLE 5 (neutral polarity → mentioned_only)
Review: "The menu lists vegetarian options."
Aspect: menu
Polarity: neutral
Correct Emotion: mentioned_only
Reason: "The reviewer simply states factual menu information without expressing any emotional stance, meaning the mention carries no emotional depth or subjective evaluation."

### EXAMPLE 6 (neutral polarity → no_emotion)
Review: "There are tables near the back of the restaurant."
Aspect: place
Polarity: neutral
Correct Emotion: no_emotion
Reason: "The description of seating location is presented factually without any emotional meaning, indicating no emotional content associated with the specific aspect described."

==============================
END OF FEW-SHOT GUIDELINES
==============================
"""