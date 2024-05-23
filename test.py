import sys

def main():
    ex_model = sys.argv[1]
    num = sys.argv[2]  # The fixed number 4
    ab_model = sys.argv[3]
    mode = sys.argv[5]  # sys.argv[4] is '-m'

    print(f"EX_MODEL: {ex_model}, NUM: {num}, AB_MODEL: {ab_model}, MODE: {mode}")

if __name__ == "__main__":
    main()
    print("Hello World!")