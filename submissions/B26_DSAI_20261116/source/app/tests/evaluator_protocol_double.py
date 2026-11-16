"""CI-only evaluator PROTOCOL double. Not official scoring and never a model result.
The external official evaluator is exercised separately when supplied by the user.
"""
import json
import sys
VALUE_SIM=.85
SINGLE_VALUED={'service_state','tariff','ticket_status'}

def _value_sim(a,b):return float(a==b)

def match_facts(gold,pred):
    used=set();pairs=[]
    for g in gold:
        for j,p in enumerate(pred):
            if j not in used and all(g[k]==p[k] for k in ['attribute','value','polarity']):
                used.add(j);pairs.append((g,p,1.0));break
    return pairs

def evaluate(gold,pred):
    ng=sum(len(g['facts']) for g in gold.values());np=sum(len(p.get('facts',[])) for p in pred.values())
    pairs=[pair for cid,g in gold.items() for pair in match_facts(g['facts'],pred.get(cid,{}).get('facts',[]))]
    matched=len(pairs)
    return {'n_calls':len(gold),'n_gold_facts':ng,'n_pred_facts':np,
        'fact_precision':matched/np if np else 0.,'fact_recall':matched/ng if ng else 0.,
        'temporal_recall':1. if matched else None,'state_accuracy':None,'n_state':0,
        'grounding_turn':1. if matched else None,'grounding_span':None,'n_temporal':matched,'n_grounded':matched}

if __name__=='__main__':
    gold={r['call_id']:r for r in map(json.loads,open(sys.argv[1]))}
    pred={r['call_id']:r for r in map(json.loads,open(sys.argv[2]))}
    json.dump(evaluate(gold,pred),open(sys.argv[4],'w'))
