import re
import unittest
from reproduction.grounding_eval.replacements import make_replacements, _norm, _roman_int

class TypedCounterfactualReplacement(unittest.TestCase):
    def check(self,golds,question):
        x=make_replacements(golds,['nq',17],question)
        self.assertEqual(x,make_replacements(list(reversed(golds)),['nq',17],question))
        self.assertEqual(set(x['alias_to_value']),set(golds))
        self.assertEqual(set(x['cf_aliases']),set(x['alias_to_value'].values()))
        self.assertTrue(all(_norm(v)not in {_norm(g)for g in golds}for v in x['cf_aliases']))
        self.assertIn(x['injected_value'],x['cf_aliases'])
        return x
    def test_named_dates_preserve_month_day_and_shared_year(self):
        x=self.check(['May 18, 2018','18 May 2018','2018'],'When was the event?')
        m=x['alias_to_value'];year=m['2018']
        self.assertEqual(m['May 18, 2018'],'May 18, '+year)
        self.assertEqual(m['18 May 2018'],'18 May '+year)
        self.assertEqual(x['replacement_kind'],'date_preserved_year_offset')
    def test_compact_date_and_iso_do_not_become_entities(self):
        x=self.check(['May18,2018','2018-05-18'],'What was the date?')
        self.assertRegex(x['alias_to_value']['May18,2018'],r'^May18,\d{4}$')
        self.assertRegex(x['alias_to_value']['2018-05-18'],r'^\d{4}-05-18$')
    def test_leap_day_remains_a_valid_date(self):
        import calendar
        for key in range(100):
            x=make_replacements(['February 29, 2020','2020-02-29'],key,'When?')
            for value in x['cf_aliases']:
                year=int(re.search(r'[12]\d{3}',value).group())
                self.assertTrue(calendar.isleap(year))

    def test_percentage_keeps_percent_and_numeric_alias(self):
        x=self.check(['20%','20'],'What percentage?');m=x['alias_to_value']
        self.assertEqual(m['20%'],m['20']+'%')
        self.assertGreaterEqual(int(m['20']),0);self.assertLessEqual(int(m['20']),100)
        self.assertEqual(x['replacement_kind'],'percentage_preserved')
    def test_count_unit_aliases_share_new_number(self):
        x=self.check(['291 episodes','291'],'How many episodes?');m=x['alias_to_value']
        self.assertEqual(m['291 episodes'],m['291']+' episodes')
        self.assertEqual(x['replacement_kind'],'numeric_quantity_unit_preserved')
    def test_decimal_units_and_comma_form(self):
        x=self.check(['1,200.5 km','1200.5'],'How long?');m=x['alias_to_value']
        self.assertEqual(m['1,200.5 km'].replace(',',''),m['1200.5']+' km')
    def test_roman_season_pair_changes_same_offset(self):
        x=self.check(['Super Bowl LII','2017'],'Who won Super Bowl LII in the 2017 season?')
        m=x['alias_to_value'];roman=m['Super Bowl LII'].split()[-1]
        self.assertEqual(_roman_int(roman)-52,int(m['2017'])-2017)
        self.assertEqual(x['replacement_kind'],'roman_and_season_year_offset')
    def test_question_pools_are_semantically_coarse_but_not_cross_type(self):
        tests=[(['Barack Obama','Obama'],'Who was president?','person'),
               (['Paris'],'In which city was it held?','location'),
               (['Germany'],'Which country borders it?','country'),
               (['temporal lobe'],'Which brain lobe handles hearing?','anatomy_lobe'),
               (['pituitary gland'],'Which gland produces it?','anatomy_gland'),
               (['hippocampus'],'Where in the brain?','anatomy_brain'),
               (['liver'],'Where in the body?','anatomy_body')]
        for golds,q,kind in tests:
            x=self.check(golds,q);self.assertEqual(x['replacement_kind'],'question_hint_'+kind)
    def test_unknown_entity_type_is_marked(self):
        x=self.check(['An obscure object'],'What is this called?')
        self.assertEqual(x['replacement_kind'],'fallback_entity_unknown_type')
    def test_gold_collision_avoided_for_actual_values(self):
        for i in range(40):
            x=make_replacements(['291 episodes','291','292'],['nq',i],'How many episodes?')
            self.assertTrue(all(_norm(v)not in {'291 episodes','291','292'}for v in x['cf_aliases']))

if __name__=='__main__':unittest.main()
