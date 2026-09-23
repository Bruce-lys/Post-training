"""Dump MegatronSftArguments dataclass fields (default / type / choices / help) to JSON.
Run with the TRAINING python (.venv-swift), e.g. scripts/export_schema.sh. The UI process only reads the JSON."""
import dataclasses, json, re, sys, typing

def _literal_choices(t):
    if isinstance(t, str):
        m = re.search(r"Literal\[(.*?)\]", t)
        if m:
            return [x.strip().strip("'\"") for x in m.group(1).split(",")]
        return None
    origin = typing.get_origin(t)
    if origin is typing.Literal:
        return list(typing.get_args(t))
    if origin is typing.Union:
        for a in typing.get_args(t):
            c = _literal_choices(a)
            if c:
                return c
    return None

def main(out_path):
    from swift.megatron.arguments import MegatronSftArguments
    out = {}
    for f in dataclasses.fields(MegatronSftArguments):
        if f.default is not dataclasses.MISSING:
            d = f.default
        elif f.default_factory is not dataclasses.MISSING:
            d = f.default_factory()
        else:
            d = None
        try:
            json.dumps(d)
        except Exception:
            d = str(d)
        help_ = None
        if f.metadata:
            help_ = f.metadata.get("help")
        out[f.name] = {"default": d, "type": str(f.type), "choices": _literal_choices(f.type), "help": help_}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"wrote {len(out)} fields -> {out_path}")

if __name__ == "__main__":
    main(sys.argv[1])
