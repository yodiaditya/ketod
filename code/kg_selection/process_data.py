# Copyright (c) Meta Platforms, Inc. and its affiliates.

# generate non-delex data

# Non-delex data
# "I would like to book a flight from New York to Los Angeles on October 10th."
#
# Delexicalized User Utterance:
# "I would like to book a flight from [city] to [city] on [date]."


import json
import copy
import random
import argparse
import os
import sys

parent_folder = os.path.abspath("../..")
root = parent_folder + "/ketod_release/"

def main():
    parser = argparse.ArgumentParser()

    # by default, refer to train_final.json
    parser.add_argument("--data", default="train_final.json", type=str, required=False, help="data file name")
    args = parser.parse_args()

    with open(root + args.data, "r", encoding='utf8') as f:
        data = json.load(f)

    i = 0

    # iterate over each dialogue
    while i < len(data):
        # iterate over each turn and skip on even turn. J will produce 1, 3, 5, etc
        for j in range(1, len(data[i]["turns"]), 2):
            # get system turn, service": "Travel_1" -> "travel"
            # ["frames"][0] is action in the dataset

            # Example:
            # {
            #     "frames": [
            #         {
            #             "actions": [
            #                 {
            #                     "act": "REQUEST",
            #                     "canonical_values": [],
            #                     "slot": "location",
            #                     "values": []
            #                 }
            #             ],
            #             "service": "Travel_1",
            #             "slots": []
            #         }
            #     ],
            #     "speaker": "SYSTEM",
            #     "utterance": "Sure, where are you looking?",
            #     "enrich": false
            # },
            domain = data[i]["turns"][j]["frames"][0]["service"].split("_")[0].lower() 

            # double check if its speaker is system
            assert(data[i]["turns"][j]["speaker"] == "SYSTEM")

            # make sure there is only one frame
            assert(len(data[i]["turns"][j]["frames"]) == 1)

            # example slots: [{'exclusive_end': 67, 'slot': 'attraction_name', 'start': 56}]
            slots = copy.deepcopy(data[i]["turns"][j]["frames"][0]["slots"])

            # make sure the multiple slots value with start is ordered by start
            # "slots": [
            #     {
            #         "exclusive_end": 36,
            #         "slot": "check_in_date",
            #         "start": 24
            #     },
            #     {
            #         "exclusive_end": 93,
            #         "slot": "hotel_name",
            #         "start": 56
            #     }
            # ],
            slots.sort(key = lambda x : -x["start"])

            # data[i]["turns"][j]["delex"] = delex  -> no delex
    
            # this will contain belief and action
            target = ''

            # generate the belief from user utterance turn
            belief = []

            # iterate over frames just in case it has multiple frames
            for k in range(len(data[i]["turns"][j-1]["frames"])):
                # iterate slot values
                # "actions": [
                #     {
                #         "act": "INFORM",
                #         "canonical_values": [
                #             "2019-03-07"
                #         ],
                #         "slot": "visit_date",
                #         "values": [
                #             "the 7th"
                #         ]
                #     },
                #     {
                #         "act": "NEGATE",
                #         "canonical_values": [],
                #         "slot": "",
                #         "values": []
                #     }
                # ],
                # "service": "Homes_1",
                # "slots": [
                #     {
                #         "exclusive_end": 37,
                #         "slot": "visit_date",
                #         "start": 30
                #     }
                # ],
                # "state": {
                #             "active_intent": "ScheduleVisit",
                #             "requested_slots": [],
                #             "slot_values": {
                #                 "area": [
                #                     "Petaluma"
                #                 ],
                #                 "number_of_beds": [
                #                     "2"
                #                 ],
                #                 "property_name": [
                #                     "Addison Ranch"
                #                 ],
                #                 "visit_date": [
                #                     "the 7th"
                #                 ]
                #             }
                #         }
                for slot in data[i]["turns"][j-1]["frames"][k]["state"]["slot_values"]:
                    # generate belief which contains service, slot are number_of_beds, property_name, visit_date
                    belief += [[data[i]["turns"][j-1]["frames"][k]["service"].split("_")[0].lower(), slot, data[i]["turns"][j-1]["frames"][k]["state"]["slot_values"][slot]]]
            
            # sort belief that made from each slot in slot_values
            belief.sort(key = lambda x : x[0] + " " + x[1])
    
            for k in range(len(belief)):
                belief[k][2].sort()
                belief[k][2] = belief[k][2][0]

            belief = [x[0] + " " + x[1] + " " + x[2] for x in belief]
            if len(belief) > 0:
                target += ('<|belief|> ' + ", ".join(belief) + ' <|endofbelief|> ')
            else:
                target += ('<|belief|> <|endofbelief|> ')

            action = copy.deepcopy(data[i]["turns"][j]["frames"][0]["actions"])
            action.sort(key = lambda x : x["act"])

            # db result, use gold db results
            db_list = []
            for k in range(len(data[i]["turns"][j]["frames"])):
                # gold db not found in dataset, this line will not executed
                if "gold_db" in data[i]["turns"][j]["frames"][k] and len(data[i]["turns"][j]["frames"][k]["gold_db"]) > 0:
                    # first_entry = data[i]["turns"][j]["frames"][k]["gold_db"]
                    db_list = []
                    for ind, first_entry in enumerate(data[i]["turns"][j]["frames"][k]["gold_db"]):
                        this_db_list = [domain + " " + x + " " + first_entry[x] for x in first_entry]
                        this_db_list = ["result " + str(ind)] + this_db_list
                        db_list.extend(this_db_list)

                # "service_results": [
                #             {
                #                 "airlines": "Alaska Airlines",
                #                 "departure_date": "2019-03-04",
                #                 "destination_airport": "Kingsford Smith International Airport",
                #                 "destination_city": "Sydney",
                #                 "number_stops": "1",
                #                 "origin_airport": "SAN International Airport",
                #                 "origin_city": "San Diego",
                #                 "outbound_arrival_time": "12:22",
                #                 "outbound_departure_time": "06:45",
                #                 "passengers": "2",
                #                 "price": "245",
                #                 "refundable": "False",
                #                 "seating_class": "Economy"
                #             }
                #         ],

                # check if service_results is exists in SYSTEM turn
                if "service_results" in data[i]["turns"][j]["frames"][k] and len(data[i]["turns"][j]["frames"][k]["service_results"]) > 0:
                    # count how many service results
                    num_results = len(data[i]["turns"][j]["frames"][k]["service_results"])
                    
                    # append with perivous service results if any
                    db_list = [str(num_results) + " results"] + db_list
    
            ### use the first value?
            action_list = []
            for x in action:
                if len(x["values"]) > 0:
                    action_list.append(domain + " " + x["act"].lower() + " " + x["slot"]+ " " + " ".join(x["values"]))
                else:
                    action_list.append(domain + " " + x["act"].lower() + " " + x["slot"])

            if len(db_list) > 0:
                target += ('<|dbresults|> ' + ", ".join(db_list) + ' <|endofdbresults|> ')
            else:
                # no need to retrieve new db
                target += ('<|dbresults|> <|nonewdb|> <|endofdbresults|> ')

            target += ('<|action|> ' + ", ".join(action_list) + ' <|endofaction|>')

            # # knowledge snippets '<|chitchat|>', '<|nochitchat|>', '<|knowledge|>', '<|endofknowledge|>'
            # if data[i]["turns"][j]["enrich"]:
            #     kg_text = " ".join(data[i]["turns"][j]["kg_snippets_text"])
            #     target += ('<|chitchat|> <|endofdecision|> <|knowledge|> ' + kg_text + ' <|endofknowledge|> ')
            #     this_response = data[i]["turns"][j]["enriched_utter"]
            # else:
            #     target += '<|nochitchat|> <|endofdecision|>'
            #     this_response = data[i]["turns"][j]["utterance"]
            # target += ('<|response|> ' + this_response + ' <|endofresponse|>')
            # target = target

            # data[i]["turns"][j]["target"] = target

            # generate context and included from previous as well. <|context|><|user|> user utterances <|system|> system utterances <|endofcontext|> 
            context = '<|context|> '
            for m in range(j):
                if m % 2 == 0:
                    context += '<|user|> '
                else:
                    context += '<|system|> '
                context += data[i]["turns"][m]["utterance"] + " "
            context += '<|endofcontext|> '
    
            # full_train_input = (context + target).replace("\n", " ").replace("\r", "")

            tillaction = context + target

            # # oracle test
            # context += ('<|belief|> ' + ", ".join(belief) + ' <|endofbelief|> ').lower()
            # context += ('<|dbresults|> ' + ", ".join(db_list) + ' <|endofdbresults|> ')

            context = context.strip()

            # context contain user utterances and system utterances
            # tillaction is contain context and target, which target contain belief, dbresults, and action
            data[i]["turns"][j]["context"] = context
            data[i]["turns"][j]["tillaction"] = tillaction

            print("Context: ", context)
            print("Target: ", target)
    
            # inlm += [full_train_input]
            # assert("\n" not in inlm[-1])
            # inlme += [(context).replace("\n", " ").replace("\r", "")]

        i += 1

    tgt = root + "processed_kg_select_" + args.data

    with open(tgt, "w") as f:
        json.dump(data, f, indent=1)

    # Example

    # Context:  <|context|> <|user|> I'm looking for some interesting attractions to visit, specifically a Museum. Can you help me find any? <|system|> Where should I look? <|user|> Look for some attractions in London, please. <|system|> I found a Museum called Apsley House that you should check out. <|user|> Cool, can you find me some hotels around that area? <|system|> Sure thing. I found 10 hotels that you might be interested in. How about staying at a 5 star hotel called 45 Park Lane? <|user|> What about any others? <|system|> Yeah, I found a 3 star hotel called Abc Hyde Park Hotel. <|user|> Any other hotels you can find? <|system|> I found a nice 1 star hotel called Abercorn House. <|user|> Okay, sounds good. I'd like to make a reservation there. <|system|> How many days do you plan on staying for? <|user|> I want to stay there for 4 days. <|system|> What day do you want to check in? <|user|> I want to check in on the 9th of this month. <|system|> Please confirm that you want to book 1 room for 4 days at Abercorn House in London with the check in date being March 9th. <|user|> Yup, that's right. <|system|> Your room has been booked, enjoy your stay! <|user|> What's their number? <|system|> You can reach them at +44 20 8563 8692. <|user|> Great, thanks! <|system|> Is there anything else I can help you with? <|user|> No thanks, that's all I needed. <|endofcontext|>
    # Target:  <|belief|> hotels check_in_date 9th of this month, hotels destination London, hotels hotel_name Abercorn House, hotels number_of_days 4, hotels number_of_rooms 1 <|endofbelief|> <|dbresults|> <|nonewdb|> <|endofdbresults|> <|action|> hotels goodbye  <|endofaction|>

    # Context:  <|context|> <|user|> I'm looking for some interesting attractions to visit, specifically a Museum. Can you help me find any? <|system|> Where should I look? <|user|> Look for some attractions in London, please. <|endofcontext|>
    # Target:  <|belief|> travel category Museum, travel location London <|endofbelief|> <|dbresults|> 10 results <|endofdbresults|> <|action|> travel offer attraction_name Apsley House, travel offer category Museum <|endofaction|>


if __name__ == '__main__':
    random.seed(42)
    main()
