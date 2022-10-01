import json

f = open('scenario_list.json')
scenario_list = json.load(f)

print(len(scenario_list["non-interactive"][0]))
print(len(scenario_list["interactive"][0]))
print(len(scenario_list["obstacle"][0]))
print(len(scenario_list["collision"][0]))

