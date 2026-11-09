import itertools
import math
import random
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction
from decodelab.decode import DecodingConfig, _filtered, beam_trace
from decodelab.filtering import sampling_trace
from decodelab.paths import enumerate_paths
from decodelab.model import NGramModel

class Table:
    rows = {'start': {'a': .60, 'b': .39, '</s>': .01},
            'a': {'a': .34, 'b': .33, '</s>': .33},
            'b': {'a': .01, 'b': .01, '</s>': .98}}
    def distribution(self, context):
        return self.rows[context[-1] if context else 'start']
    def unigram_count(self, token):
        return 0

def decimal_reference(dist, config):
    # Independent unshifted high precision powers, not production log/exp.
    with localcontext() as ctx:
        ctx.prec = 90
        values = sorted(((t, Decimal(str(p)) ** (1/Decimal(str(config.temperature))))
                         for t,p in dist.items() if p > 0), key=lambda x:(-x[1],x[0]))
        if config.top_k is not None:
            values = values[:config.top_k]
        z = sum(p for _,p in values)
        values = [(t,p/z) for t,p in values]
        kept=[]
        for item in values:
            if config.top_p is not None and kept and sum(p for _,p in kept) >= Decimal(str(config.top_p)):
                break
            kept.append(item)
        total=sum(p for _,p in kept)
        return {t:float(p/total) for t,p in kept}

class TestStableSampling(unittest.TestCase):
    def test_normalized_nucleus_regression(self):
        d={'a':.6,'b':.3,'c':.1}
        # At T=.5: (.36,.09,.01)/.46. a alone exceeds .7.
        self.assertEqual(_filtered(d,DecodingConfig(temperature=.5,top_p=.7)), {'a':1})
        # Top-k first renormalizes (.6,.3) -> (2/3,1/3).
        self.assertEqual(_filtered(d,DecodingConfig(top_k=2,top_p=.65)), {'a':1})
    def test_extreme_temperature_no_all_underflow(self):
        for t in [5e-324,1e-300,.001,1e300]:
            q=_filtered({'a':.6,'b':.4},DecodingConfig(temperature=t))
            self.assertAlmostEqual(sum(q.values()),1)
            self.assertTrue(all(math.isfinite(p) for p in q.values()))
    def test_equal_ties_and_exact_boundary(self):
        self.assertEqual(_filtered({'b':.5,'a':.5},DecodingConfig(top_p=.5)),{'a':1})
        self.assertEqual(set(_filtered({'b':.5,'a':.5},DecodingConfig(top_p=1))),{'a','b'})
    def test_zero_support_not_resurrected(self):
        self.assertEqual(_filtered({'a':1,'b':0},DecodingConfig(temperature=100)),{'a':1})
    def test_decimal_independent_random_sweep(self):
        rng=random.Random(708)
        for case in range(120):
            weights=[rng.randint(1,1000) for _ in range(rng.randint(2,9))]
            d={str(i):w/sum(weights) for i,w in enumerate(weights)}
            cfg=DecodingConfig(temperature=[.1,.5,.7,1,2,10][case%6],top_k=[None,1,3,20][case%4],top_p=[None,.4,.7,.95,1][case%5])
            expected=decimal_reference(d,cfg);actual=_filtered(d,cfg)
            self.assertEqual(set(actual),set(expected))
            for t in actual:self.assertAlmostEqual(actual[t],expected[t],places=13)
    def test_witness_minimality(self):
        for p in [.01,.5,.8,.99,1]:
            w=sampling_trace({'a':.5,'b':.3,'c':.2},DecodingConfig(temperature=.7,top_k=2,top_p=p))
            self.assertGreaterEqual(w['nucleus_retained_top_k_mass']+1e-15,p)
            if p<1:self.assertLess(w['nucleus_prefix_mass_before_last'],p)
    def test_bad_domains(self):
        for kw in [{'temperature':float('nan')},{'temperature':float('inf')},{'temperature':True},{'max_tokens':True},{'max_tokens':1.5},{'top_k':1.5},{'top_p':float('nan')}]:
            with self.assertRaises(ValueError):DecodingConfig(**kw)
        for d in [{},{'a':.5},{'a':float('nan')},{'a':-.1,'b':1.1},{'a':True},{'':1}]:
            with self.assertRaises(ValueError):_filtered(d,DecodingConfig())

class TestPathAudit(unittest.TestCase):
    def test_independent_rational_cartesian_tree(self):
        # Fixed products over every token string, independently absorbing at EOS.
        model=Table();cfg=DecodingConfig(max_tokens=4)
        ref={}
        for length in range(5):
            for tokens in itertools.product('ab',repeat=length):
                for stop in ['eos','cap']:
                    if (stop=='cap' and length!=4) or (stop=='eos' and length==4):continue
                    path=list(tokens)+(['</s>'] if stop=='eos' else [])
                    mass=Fraction(1);prev='start'
                    for tok in path:
                        mass*=Fraction(str(model.rows[prev][tok]));prev=tok
                    ref[(tokens,stop)]=mass
        result=enumerate_paths(model,[],cfg)
        self.assertEqual(Fraction(1),sum(ref.values()))
        self.assertEqual(len(ref),len(result['outcomes']))
        for out in result['outcomes']:
            self.assertAlmostEqual(out['probability'],float(ref[(tuple(out['tokens']),out['stop'])]),places=14)
            self.assertAlmostEqual(out['model_log_probability'],math.log(out['probability']),places=13)
        self.assertAlmostEqual(result['total_probability'],1,places=14)
    def test_sampling_policy_mass_not_model_score(self):
        r=enumerate_paths(Table(),[],DecodingConfig(max_tokens=4,temperature=.5,top_k=2,top_p=.8))
        self.assertAlmostEqual(r['total_probability'],1)
        self.assertAlmostEqual(r['eos_probability']+r['cap_probability'],1)
        self.assertTrue(any(abs(math.log(x['probability'])-x['model_log_probability'])>.1 for x in r['outcomes']))
    def test_cap_has_no_eos_factor(self):
        r=enumerate_paths(Table(),[],DecodingConfig(max_tokens=1))
        self.assertEqual(len(r['outcomes']),3)
        a=next(x for x in r['outcomes'] if x['tokens']==['a'])
        self.assertEqual(a['stop'],'cap');self.assertAlmostEqual(a['probability'],.6)
    def test_beam_pruning_counterexample(self):
        cfg=DecodingConfig(max_tokens=2)
        exact=enumerate_paths(Table(),[],cfg)['outcomes'][0]
        self.assertEqual((exact['tokens'],exact['stop']),(['b'],'eos'))
        narrow=beam_trace(Table(),[],1,cfg);wide=beam_trace(Table(),[],2,cfg)
        self.assertEqual(narrow['beams'][0]['tokens'],['a','a'])
        self.assertEqual(wide['beams'][0]['tokens'],['b'])
        self.assertTrue(wide['beams'][0]['finished'])
        self.assertAlmostEqual(math.exp(wide['beams'][0]['log_probability']),exact['probability'])
        for step in wide['steps']:
            self.assertEqual(step['survivors'],step['candidates'][:2])
    def test_prompt_conditioned_not_scored(self):
        r=enumerate_paths(Table(),['b'],DecodingConfig(max_tokens=1))
        eos=next(x for x in r['outcomes'] if x['stop']=='eos')
        self.assertAlmostEqual(eos['probability'],.98)
    def test_tree_bounds(self):
        for cfg,nodes in [(DecodingConfig(max_tokens=13),100),(DecodingConfig(max_tokens=4),2)]:
            with self.assertRaises(ValueError):enumerate_paths(Table(),[],cfg,nodes)
        with self.assertRaises(ValueError):enumerate_paths(Table(),['</s>'],DecodingConfig())
    def test_short_ngram_context_is_start_padded(self):
        m=NGramModel(3);m.train([['a','b'],['a','c']])
        self.assertEqual(m.distribution([]),m.distribution(['<s>','<s>']))
        self.assertEqual(m.distribution(['a']),m.distribution(['<s>','a']))
        self.assertGreater(m.distribution([])['a'],m.distribution([])['b'])
    def test_model_bad_domains(self):
        for order,k in [(True,.1),(1.5,.1),(2,float('nan')),(2,float('inf'))]:
            with self.assertRaises(ValueError):NGramModel(order,k)

class TestFullAudit(unittest.TestCase):
    def test_nonmonotone_width_counterexample(self):
        from decodelab.audit import build_audit,NONMONOTONE
        a=build_audit(NONMONOTONE)
        self.assertGreater(a['beams'][0]['trace']['beams'][0]['log_probability'],a['beams'][1]['trace']['beams'][0]['log_probability'])
        self.assertTrue(a['beams'][0]['attains_oracle_score'])
        self.assertFalse(a['beams'][1]['attains_oracle_score'])
        self.assertTrue(a['beams'][2]['attains_oracle_score'])
    def test_rational_dp_and_wide_beam_random_tables(self):
        from decodelab.audit import TableModel
        rng=random.Random(314)
        for i in range(40):
            counts={s:[rng.randint(0,20) for _ in range(3)] for s in ['start','a','b']}
            model=TableModel(counts);cfg=DecodingConfig(max_tokens=4)
            frontier={'start':Fraction(1)};eos=Fraction(0);expected=Fraction(0)
            for _ in range(4):
                nxt={'a':Fraction(0),'b':Fraction(0)}
                for state,mass in frontier.items():
                    row=counts[state];total=sum(row)
                    eos+=mass*Fraction(row[2],total)
                    for tok,n in zip('ab',row[:2]):nxt[tok]+=mass*Fraction(n,total)
                expected+=sum(nxt.values());frontier=nxt
            result=enumerate_paths(model,[],cfg)
            self.assertAlmostEqual(result['eos_probability'],float(eos),places=13)
            self.assertAlmostEqual(result['cap_probability'],float(sum(frontier.values())),places=13)
            self.assertAlmostEqual(result['expected_generated_tokens'],float(expected),places=13)
            winner=beam_trace(model,[],32,cfg)['beams'][0]
            self.assertAlmostEqual(winner['log_probability'],result['outcomes'][0]['model_log_probability'],places=13)
    def test_strict_spec_and_cli(self):
        import tempfile, pathlib,contextlib,io,json
        from decodelab.audit import read_spec
        from decodelab.cli import main
        with tempfile.TemporaryDirectory() as root:
            p=pathlib.Path(root)/'spec.json'
            for text in ['{"counts": {}, "counts": {}}','{"unknown": 1}','{"config":{"temperature":NaN}}','{"counts":{"start":[1,2,3],"a":[1,2,3],"b":[0,0,0]}}']:
                p.write_text(text)
                with self.assertRaises((ValueError,TypeError)):read_spec(p)
            p.write_text('{"config":{"max_tokens":2,"temperature":0.5,"top_p":0.7}}')
            out=pathlib.Path(root)/'out.json';buf=io.StringIO()
            with contextlib.redirect_stdout(buf):code=main(['audit','--spec',str(p),'--json',str(out)])
            self.assertEqual(code,0);self.assertEqual(json.loads(buf.getvalue()),json.loads(out.read_text()))
            self.assertEqual(json.loads(out.read_text())['schema'],'decodelab-audit-v1')
