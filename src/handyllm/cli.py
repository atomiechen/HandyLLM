import argparse
from handyllm import hprompt


def hprompt_command(args):
    prompt = hprompt.load_from(args.path)
    print(prompt.meta)
    print(prompt.data)
    # TODO

def main():
    """Main entry point for the handyllm CLI."""
    parser = argparse.ArgumentParser(description="HandyLLM CLI")
    subparsers = parser.add_subparsers(dest="command")
    parser_hprompt = subparsers.add_parser('hprompt', help="Process hprompt files")
    parser_hprompt.add_argument("path", help="Path to hprompt file")
    args = parser.parse_args()
    if args.command == "hprompt":
        return hprompt_command(args)


if __name__ == "__main__":
    main()

