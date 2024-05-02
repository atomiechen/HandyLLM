import argparse
from handyllm import hprompt


def register_hprompt_command(subparsers):
    parser_hprompt = subparsers.add_parser('hprompt', help="Run hprompt files")
    parser_hprompt.add_argument("path", help="Path to hprompt file")
    parser_hprompt.add_argument("-o", "--output", help="Output path")
    parser_hprompt.add_argument("-v", "--verbose", action="store_true", default=False, help="Verbose output")

def hprompt_command(args):
    if args.verbose:
        print(f"Input path: {args.path}")
        print(f"Output path: {args.output}")
    prompt = hprompt.load_from(args.path)
    result = prompt.run()
    if args.output:
        result.dump_to(args.output)
    else:
        print(result.data)

def main():
    """Main entry point for the handyllm CLI."""
    parser = argparse.ArgumentParser(description="HandyLLM CLI")
    subparsers = parser.add_subparsers(dest="command")
    register_hprompt_command(subparsers)
    args = parser.parse_args()
    if args.command == "hprompt":
        return hprompt_command(args)


if __name__ == "__main__":
    main()

