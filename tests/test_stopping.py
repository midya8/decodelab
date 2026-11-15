"""Cartesian path products and closed-form controls, independent of state flow."""
import copy
from fractions import Fraction as F
import itertools
import random
import unittest
from decodelab.stopping import stopping_law


def cartesian(counts, prompt, cap):
    law = [F(0)] * (cap+1)
    for length in range(cap+1):
        for path in itertools.product('ab', repeat=length):
            mass = F(1)
            state = prompt[-1] if prompt else 'start'
            for token in path:
                row = counts[state]
                mass *= F(row['ab'.index(token)], sum(row))
                state = token
            if length < cap:
                row = counts[state]
                mass *= F(row[2], sum(row))
            law[length] += mass
    return law


class StoppingTests(unittest.TestCase):
    def replay(self, report):
        frontier = {report['prompt'][-1] if report['prompt'] else 'start': F(1)}
        law = []
        for depth, step in enumerate(report['steps']):
            self.assertEqual(step['generated_before'], depth)
            self.assertEqual({k:F(v) for k,v in step['incoming'].items()}, frontier)
            self.assertEqual(F(step['incoming_mass']), sum(frontier.values()))
            expected_keys = {(state,token) for state in frontier for token in ('a','b','</s>')}
            self.assertEqual({(t['state'],t['token']) for t in step['transitions']}, expected_keys)
            self.assertEqual(len(step['transitions']), len(expected_keys))
            outgoing = {'a':F(0),'b':F(0)}
            stopped = F(0)
            for transition in step['transitions']:
                state, token = transition['state'], transition['token']
                weights = report['counts'][state]
                p = F(weights[('a','b','</s>').index(token)],sum(weights))
                self.assertEqual(F(transition['probability']),p)
                flow = frontier[state]*p
                self.assertEqual(F(transition['mass']),flow)
                if token == '</s>': stopped += flow
                else: outgoing[token] += flow
            self.assertEqual(sum(outgoing.values())+stopped, sum(frontier.values()))
            self.assertEqual({k:F(v) for k,v in step['next_states'].items()},outgoing)
            self.assertEqual(F(step['surviving_mass']),sum(outgoing.values()))
            self.assertEqual(F(step['eos_mass']),stopped)
            law.append(stopped)
            frontier=outgoing
        law.append(sum(frontier.values()))
        self.assertEqual(law,list(map(F,report['length_probability'])))
        self.assertEqual(sum(law),1)
        mean=sum(i*p for i,p in enumerate(law))
        self.assertEqual(F(report['expected_generated_tokens']),mean)
        self.assertEqual(F(report['variance_generated_tokens']),sum((i-mean)**2*p for i,p in enumerate(law)))
        self.assertEqual(mean,sum(F(step['surviving_mass']) for step in report['steps']))
        self.assertEqual(F(report['cap_probability']),law[-1])
        self.assertEqual(F(report['eos_probability']),sum(law[:-1]))

    def test_random_integer_cartesian_laws_and_full_flow_replay(self):
        rng=random.Random(1142)
        for case in range(48):
            counts={state:[rng.randint(0,30) for _ in range(3)] for state in ('start','a','b')}
            for row in counts.values():
                if not sum(row): row[0]=1
            for prompt in ([],['a'],['b','a','b']):
                cap=1+case%7
                result=stopping_law(counts,prompt,cap)
                self.assertEqual(list(map(F,result['length_probability'])),cartesian(counts,prompt,cap))
                self.replay(result)

    def test_geometric_cap_128_no_invented_eos(self):
        counts={s:[2,3,5] for s in ('start','a','b')}
        for cap in (1,2,12,128):
            r=stopping_law(counts,[],cap)
            expected=[F(1,2)**(n+1) for n in range(cap)]+[F(1,2)**cap]
            self.assertEqual(list(map(F,r['length_probability'])),expected)
            self.replay(r)
        # Cap1 counts only the first transition: EOS0.5, one emitted token0.5.
        self.assertEqual(stopping_law(counts,[],1)['length_probability'],['1/2','1/2'])

    def test_zero_support_immediate_eos_and_perpetual_live_mass(self):
        for counts,expected in [({s:[0,0,1] for s in ('start','a','b')},0),
                                ({s:[1,0,0] for s in ('start','a','b')},128)]:
            r=stopping_law(counts,[],128);self.replay(r)
            self.assertEqual(F(r['expected_generated_tokens']),expected)
            self.assertEqual(F(r['variance_generated_tokens']),0)
        counts={'start':[0,1,0],'a':[0,0,1],'b':[1,0,0]}
        r=stopping_law(counts,[],4)
        self.assertEqual(list(map(F,r['length_probability'])),[0,0,1,0,0])
        self.replay(r)

    def test_row_scaling_prompt_conditioning_and_no_input_aliasing(self):
        counts={'start':[1,2,3],'a':[5,0,7],'b':[0,2,1]}
        prompt=['b','a'];before=copy.deepcopy((counts,prompt))
        a=stopping_law(counts,prompt,8)
        b=stopping_law({k:[n*(i+2) for n in row] for i,(k,row) in enumerate(counts.items())},prompt,8)
        for key in ('steps','length_probability','eos_probability','cap_probability','expected_generated_tokens','variance_generated_tokens'):
            self.assertEqual(a[key],b[key])
            self.assertEqual(a[key],stopping_law(counts,['a'],8)[key])
        self.assertEqual((counts,prompt),before)
        a['counts']['a'][0]=999;a['prompt'].append('b')
        self.assertEqual((counts,prompt),before)

    def test_rejects_domains_before_returning_partial_evidence(self):
        for cap in (True,0,129,1.5,None):
            with self.assertRaises(ValueError):stopping_law(cap=cap)
        for prompt in ('a',('a',),['</s>'],['<s>'],['z'],[False]):
            with self.assertRaises(ValueError):stopping_law(prompt=prompt)
        for counts in ({}, {'start':[1,1,1],'a':[1,1,1],'b':[0,0,0]},
                       {'start':[1,1,1],'a':[True,1,1],'b':[1,1,1]}):
            with self.assertRaises(ValueError):stopping_law(counts)


if __name__=='__main__':unittest.main()
