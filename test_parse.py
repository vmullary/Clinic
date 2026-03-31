import re

def parse_prediction(text, persons, pets):
    person_room = {}
    pet_room = {}
    
    # 1. Isolate the final conclusion
    text_lower = text.lower()
    keywords = ["final answer", "final configuration", "the tenants are", "conclusion:", "tenants:"]
    block = text_lower
    for kw in keywords:
        if kw in text_lower:
            block = text_lower[text_lower.rfind(kw):]
            break
            
    # Remove negative sentences
    sentences = re.split(r'[\n\.]', block)
    positive_sentences = []
    for s in sentences:
        if not re.search(r'\b(not|cannot|can\'t|n\'t)\b', s):
            positive_sentences.append(s)
            
    # Map entities sentence by sentence
    for s in positive_sentences:
        rooms = re.findall(r'(?:room\s*|r)(\d)\b', s)
        if not rooms:
            continue
            
        # If there's ambiguous rooms in the same segment, skip or take the first
        # Usually it's one room per sentence like "Room 1 has Alice and cat"
        # Or "Charlie is in room 3"
        room_num = int(rooms[0])
        
        for p in persons:
            if re.search(r'\b' + re.escape(p.lower()) + r'\b', s) and p not in person_room:
                person_room[p] = room_num
                
        for pt in pets:
            if re.search(r'\b' + re.escape(pt.lower()) + r'\b', s) and pt not in pet_room:
                pet_room[pt] = room_num

    return person_room, pet_room

text1 = """
The final answer is:
Room 1 has Alice and her cat.
Room 2: Bob, dog.
Room 3: Charlie and bat.
"""
print(text1.strip(), "->", parse_prediction(text1, ["Alice", "Bob", "Charlie"], ["cat", "dog", "bat"]))

text2 = """
Let's figure this out.
Alice is not in 2.
Final Configuration:
room 1: Alice & dog
room 2: Bob & cat
"""
print(text2.strip(), "->", parse_prediction(text2, ["Alice", "Bob", "Charlie"], ["cat", "dog", "bat"]))

text3 = """
Room 1 has Alice. And Room 1 has cat.
Room 2 is where Bob is, with the dog.
"""
print(text3.strip(), "->", parse_prediction(text3, ["Alice", "Bob", "Charlie"], ["cat", "dog", "bat"]))
