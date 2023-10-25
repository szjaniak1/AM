import parser as par

import sys

def main() -> int:
	par.parse(str(sys.argv[1]))

if __name__ == '__main__':
    main()