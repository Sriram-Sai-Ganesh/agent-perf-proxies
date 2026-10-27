"""Check the JVM starts and a gold path reaches full score: python -m env_scienceworld.smoke"""

import time

from env_scienceworld.adapter import ScienceWorld

t = time.time()
env = ScienceWorld()
obs = env.reset("scienceworld/boil/0")
gold = env.env.get_gold_action_sequence()  # gold needs generateGoldPath; reset skips it, so load again
env.env.load("boil", 0, env.simplification, generateGoldPath=True)
env.env.reset()
gold = list(env.env.get_gold_action_sequence())
res = None
for a in gold:
    res = env.step(a)
    if res.done:
        break
assert res.score == 1.0 and res.done, (res.score, res.done)
assert not env.step("fly to the moon").valid
env.close()
print(f"ok: boil/0 gold path ({len(gold)} steps) reaches score 1.0 in {time.time() - t:.1f}s")
