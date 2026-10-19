import json
import subprocess
import sys
from pathlib import Path

import pytest

from reproduction.grounding_eval.core_v5 import PROTOCOL, aggregate, classify, intervene, prepare_row
from reproduction.grounding_eval.replacements_v5 import make_plan, parse_quantity
from reproduction.grounding_eval.strict_v5 import validated


def test_raw_decimal_and_currency_not_qa_normalized():
    p=make_plan(['$2.187 billion'],1729,'how much did it make')
    assert p['status']=='VALID_PLAN'
    assert list(p['alias_to_value'].values())[0].startswith('$')
    assert parse_quantity('$2.187 billion')['value']==2187000000
    assert make_plan(['$2.18 billion','$2.187 billion'],1729,'how much')['status']=='INVALID_INTERVENTION'


def test_numeric_whole_answer_equivalence():
    assert classify('seven','7',['12'],['seven'])=='Hall'
    assert classify('seven','twelve',['12'],['seven'])=='True'
    assert classify('seven','there are 12',['12'],['seven'])=='Amb'
    assert classify('36 months','four years old',['42 months'],['36 months'])=='Amb'
    assert classify('36 months','42 months',['42 months'],['36 months'])=='True'


def test_explicit_relative_position_equivalence():
    assert classify('beneath the liver','Under the kidney',['beneath the kidney'],['beneath the liver'])=='True'
    assert classify('beneath the liver','Under the liver',['beneath the kidney'],['beneath the liver'])=='Hall'
    assert classify('beneath the liver','near the kidney',['beneath the kidney'],['beneath the liver'])=='Amb'


def test_original_alternative_gold_is_hall_even_changed_text():
    assert classify('28 January 1813','1813',['1825'],['28 January 1813','1813'])=='Hall'


@pytest.mark.parametrize('question,golds',[
    ('when do new episodes return',['May 13, 2018','October 1, 2017']),
    ("who's playing in the world series and when",['Houston Astros','Los Angeles Dodgers','October 24 – November 1']),
    ('what are the functions of skeleton',['support','movement']),
    ('where was it filmed',['Australia','New Zealand']),
    ('author of the hymn',['William M. Runyan','Thomas Chisholm']),
    ('what year',['1960','1980']),
    ('when did it happen',['late 1970s','1974']),
    ('who won independence from whom',['the British Empire','Great Britain']),
    ('when last won',['Super Bowl LII,','2017']),
])
def test_ambiguous_entities_and_multislot_are_excluded(question,golds):
    p=make_plan(golds,1729,question)
    assert p['status']=='INVALID_INTERVENTION' and p['invalid_reason']


def test_organ_mountains_is_a_region_not_anatomical_substring():
    p=make_plan(['southern New Mexico'],1729,'where are the organ mountains in new mexico')
    assert p['replacement_kind']=='us_region'
    assert list(p['alias_to_value'].values())[0].startswith('southern ')
    assert not any('liver' in v for v in p['alias_to_value'].values())


def test_who_does_not_force_country_to_person():
    p=make_plan(['Nepal'],1729,'who uses Bikram Sambhat')
    assert p['replacement_kind']=='country'


def test_ordered_date_range_is_kept_ordered():
    p=make_plan(['6–14 July'],1729,'when are the running of the bulls')
    from reproduction.grounding_eval.replacements_v5 import parse_date
    changed=parse_date(p['alias_to_value']['6–14 July'])
    assert changed['value'][1]<=changed['value'][2]
    assert make_plan(['15–14 July'],1729,'when')['status']=='INVALID_INTERVENTION'


def test_preserves_percentage_qualifier_and_raw_punctuation():
    p=make_plan(['In 2015, 13.5%','13.5%'],1729,'what percentage')
    assert p['alias_to_value']['In 2015, 13.5%'].startswith('In 2015, ')
    assert p['alias_to_value']['13.5%'].endswith('%')


def test_only_actual_observed_injections_are_cf_aliases():
    r=intervene(['It was written in 1813.'],['28 January 1813','1813'],1729,'when written')
    assert r['status']=='READY'
    assert all('January' not in x for x in r['cf_aliases'])
    assert len(r['replacement_mapping'])==1


def test_replaces_explicit_numeric_synonyms_in_actual_docs():
    r=intervene(['There are seven units. The number is 7.'],['seven'],1729,'how many units')
    assert r['status']=='READY' and r['replacement_count']==2
    assert 'seven' not in r['modified_docs'][0].lower()


def test_no_partial_or_cascading_replacement():
    r=intervene(['Anna knows Ann Smith. Ann Smith saw Anna.'],['Ann Smith'],1729,'who is the author')
    assert r['status']=='READY' and r['replacement_count']==2
    assert r['modified_docs'][0].count('Anna')==2


def test_unresolved_surname_is_not_left_as_if_fully_intervened():
    r=intervene(['Ann Smith said yes. Smith later wrote it.'],['Ann Smith'],1729,'who is the author')
    assert r['status']=='INVALID_INTERVENTION'
    assert r['invalid_reason']=='unresolved_person_surname_residual'


def test_no_match_is_explicit_unperturbable():
    r=intervene(['Nobody is named here.'],['Ann Smith'],1729,'who is the author')
    assert r['status']=='UNPERTURBABLE'


def test_exclusions_and_denominator_are_auditable():
    p=[dict(request_id='1',status='INVALID_INTERVENTION',invalid_reason='mixed',original_em=1),
       dict(request_id='2',status='UNPERTURBABLE',invalid_reason='no_match',original_em=1)]
    s=aggregate(p,[])
    assert s['coverage']['correct_searched']==2 and s['coverage']['perturbable']==0
    assert s['TC'] is None and len(s['coverage']['invalid_reasons'])==2


def test_old_manifest_rejected_before_cache_or_model_access(tmp_path):
    f=tmp_path/'manifest.json';f.write_text(json.dumps(dict(protocol_version='paper-strict-v1')))
    with pytest.raises(ValueError,match='only accepts new strict-v5'): validated(f)


def test_v5_request_identity_differs_and_original_em_unchanged():
    row=dict(data_source='nq',evaluation_ordinal=17,qid='q',question='how many',gold=['seven'],
             model_spans=['<search>units</search>','<answer>seven</answer>'],seen_docs=['There are seven.'],
             actual_retrieval_calls=1,source_shard_sha256='abc')
    m=dict(method='G1',scale='3B',seed=None,checkpoint_hash='ckpt',cf_config_sha256='v5')
    p=prepare_row(row,m)
    assert p['original_em']==1 and p['protocol_version']==PROTOCOL and p['status']=='READY'
    from reproduction.grounding_eval.core import prepare_row as old
    assert p['request_id']!=old(row,m)['request_id']


def test_cli_is_executable():
    proc=subprocess.run([sys.executable,'-m','reproduction.grounding_eval.strict_v5','--help'],capture_output=True,text=True)
    assert proc.returncode==0 and '{prepare,run,pilot,aggregate,verify}' in proc.stdout


@pytest.mark.parametrize('text',['3.7','7.5','-7','$7','twenty-seven','7,000','3,7','+7','seven hundred'])
def test_no_numeric_subslot_match(text):
    gold='seven' if text in ('twenty-seven','seven hundred') else '7'
    result=intervene(['The answer is '+text+'.'],[gold],1729,'how many players')
    assert result['status']=='UNPERTURBABLE'


def test_standalone_number_with_unit_remains_valid():
    result=intervene(['There are 7 players.'],['7'],1729,'how many players')
    assert result['status']=='READY' and result['replacement_count']==1


def test_unsafe_numeric_subslot_not_residual_or_replaced():
    result=intervene(['There are 7 players. Value 3.7 was measured.'],['7'],1729,'how many players')
    assert result['status']=='READY' and result['replacement_count']==1
    assert '3.7' in result['modified_docs'][0]



def test_alias_family_numeric_targets_are_consistent():
    r=intervene(['In 2015, 13.5% were affected.'],['In 2015, 13.5%','13.5%'],1729,'what percentage')
    assert r['status']=='READY'
    number=next(x for x in r['cf_aliases'] if x.endswith('%') and not x.startswith('In'))
    assert classify('13.5%',number,r['cf_aliases'],r['original_semantic_aliases'])=='True'
    r=intervene(['at least seven players and seven others'],['at least seven players','at least seven','seven'],1729,'how many players')
    assert r['status']=='READY'


def test_bare_age_requires_raw_gold_license():
    r=intervene(['She was 17-year-old.'],['17-year-old','17'],1729,'how old was Baby')
    assert r['status']=='READY'
    assert any(x.isdigit() for x in r['cf_aliases'])


def test_legal_unit_equivalence_preserves_value():
    assert classify('seven','7',['13'],['seven'])=='Hall'
    assert classify('36 months','3 years old',['42 months'],['36 months'])=='Hall'
    assert classify('36 months','four years old',['42 months'],['36 months'])=='Amb'
    assert classify('20%','21 percent',['21%'],['20%'])=='True'



def test_person_relation_prefix_is_preserved():
    r=intervene(['It was written by John Smith.'],['written by John Smith'],1729,'who wrote it')
    assert r['status']=='READY'
    assert r['modified_docs'][0].startswith('It was written by ')
    assert r['replacement_mapping'][0]['injected'].startswith('written by ')
