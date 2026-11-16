"""CPU diagnostics using official matching; no extra model generation or metric replacement."""
from collections import Counter
def analyze(evaluator,gold,predictions):
    groups={};examples=[]
    for cid,call in gold.items():
        gf=call['facts'];pf=predictions.get(cid,{}).get('facts') or []
        pairs=evaluator.match_facts(gf,pf);matched_g={id(g) for g,p,s in pairs};matched_p={id(p) for g,p,s in pairs}
        for f,side,matched in [(g,'gold',id(g) in matched_g) for g in gf]+[(p,'pred',id(p) in matched_p) for p in pf]:
            attr=f['attribute'];kind='closed_history' if f.get('valid_to') else 'open_state_or_event'
            for label in [attr,kind]:
                counts=groups.setdefault(label,Counter());counts[side]+=1;counts['matched_'+side]+=matched
            if not matched and len(examples)<160:examples.append({'call_id':cid,'kind':'missed_gold' if side=='gold' else 'unmatched_prediction','fact':f})
        for g,p,_score in pairs:
            if any((g.get(k) or '')[:10]!=(p.get(k) or '')[:10] for k in ['valid_from','valid_to']) and len(examples)<160:
                examples.append({'call_id':cid,'kind':'date_mismatch_in_matched_fact','gold':g,'pred':p})
            if g['source']['turn_idx']!=p['source']['turn_idx'] and len(examples)<160:
                examples.append({'call_id':cid,'kind':'source_mismatch_in_matched_fact','gold':g,'pred':p})
    return {'scope':'Descriptive slices, not new official metrics. Temporal history separately from open states/events; attributes separately.',
        'groups':{k:dict(v) for k,v in groups.items()},'examples':examples}
