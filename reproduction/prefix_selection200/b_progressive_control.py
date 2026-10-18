"""Process-scoped B progressive prefix phases; no extra training budget."""
import argparse,json
from pathlib import Path
from .b_progressive import migrate,complete

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('migrate','complete'));p.add_argument('--base');p.add_argument('--prefix',type=int);p.add_argument('--config');p.add_argument('--outer',type=int);a=p.parse_args()
    result=migrate(a.base,a.prefix)if a.action=='migrate'else complete(a.config,a.outer)
    print(json.dumps(result))
if __name__=='__main__':main()
