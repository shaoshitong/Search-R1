"""Allocation deadline must retain finished work and cancel only pending questions."""
import asyncio
import time
from evaluate import wait_progress


async def main():
    accepted = []
    async def question(name, delay):
        await asyncio.sleep(delay)
        accepted.append(name)
        return name
    fast=asyncio.create_task(question('durable', .005))
    slow=asyncio.create_task(question('unaccepted', 30))
    deadline=time.time()+.15
    done,pending=await wait_progress({fast,slow},deadline)
    assert {x.result() for x in done}=={'durable'} and pending=={slow}
    done,pending=await wait_progress(pending,deadline)
    assert not done and pending=={slow}
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending,return_exceptions=True)
    assert accepted==['durable'] and slow.cancelled()
    # A new allocation can admit the unaccepted question once, preserving the old result.
    await question('resumed',0)
    assert accepted==['durable','resumed']
    print('ALLOCATION_DEADLINE_DURABLE_ACCEPTED_ONLY_RESUME_PASS')


if __name__=='__main__':
    asyncio.run(main())
