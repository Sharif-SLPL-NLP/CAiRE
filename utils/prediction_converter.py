import json

with open("predictions.json", 'r') as f:
    data = f.read()

data = json.loads(data)

pr = {}
for id_, val in data.items():
    x = id_.split('_')
    id = f"{x[0]}_{x[1]}"
    turn_id = int(x[2])
    if id not in pr:
        pr[id] = {
            "id": id,
            "grounding": val,
            "utterance": val,
            "turn_id": turn_id
        }
    else:
        if turn_id > pr[id]["turn_id"]:
            pr[id] = {
                "id": id,
                "grounding": val,
                "utterance": val,
                "turn_id": turn_id
            }

my_list = [value for key,value in pr.items()]

json_obj = json.dumps(my_list, indent=2)

with open("predicted.json", "w", encoding="utf-8") as f:
    f.write(json_obj)