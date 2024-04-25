import json

with open('single_round.json') as f:
    data = json.load(f)
with open('trunk_0.json','w') as f:
    json.dump(data[:100],f,ensure_ascii=False,indent=4)
with open('single_round.json','w') as f:
    json.dump(data[100:],f,ensure_ascii=False,indent=4)