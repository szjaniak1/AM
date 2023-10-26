import parser as par

import sys

def main() -> int:
	result = par.parse(str(sys.argv[1]))

	print(result)

if __name__ == '__main__':
    main()