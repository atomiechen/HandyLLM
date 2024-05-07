import argparse


def register_hprompt_command(subparsers: argparse._SubParsersAction):
    parser_hprompt = subparsers.add_parser(
        'hprompt', 
        help="Run hprompt files",
        description="Run hprompt files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_hprompt.add_argument("path", nargs='+', help="Path(s) to hprompt file")
    parser_hprompt.add_argument("-o", "--output", help="Output path; if not provided, output to stderr")
    parser_hprompt.add_argument("-v", "--verbose", action="store_true", default=False, help="Verbose output")
    parser_hprompt.add_argument("-vm", "--var-map", help="Variable map in the format key1=value1|key2=value2")
    parser_hprompt.add_argument("-vmp", "--var-map-path", help="Variable map file path")

def hprompt_command(args):
    import sys
    from handyllm import hprompt
    
    run_config = hprompt.RunConfig()
    if args.var_map:
        var_map = {}
        for pair in args.var_map.split("|"):
            key, value = pair.split("=", maxsplit=1)
            var_map[key.strip()] = value.strip()
        run_config.var_map = var_map
    if args.var_map_path:
        run_config.var_map_path = args.var_map_path
    if args.output:
        run_config.output_path = args.output
    else:
        run_config.output_fd = sys.stderr
    if args.verbose:
        run_config.verbose = True
        print(f"Input paths: {args.path}", file=sys.stderr)
    prompt = hprompt.load_from(args.path[0])
    result_prompt = prompt.run(run_config=run_config)
    for next_path in args.path[1:]:
        prompt += result_prompt
        prompt += hprompt.load_from(next_path)
        result_prompt = prompt.run(run_config=run_config)

def cli():
    """Main entry point for the handyllm CLI."""
    parser = argparse.ArgumentParser(
        prog="handyllm",
        description="HandyLLM CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")
    register_hprompt_command(subparsers)
    args = parser.parse_args()
    if args.command == "hprompt":
        return hprompt_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()

