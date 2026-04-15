from svy.errors import SvyError


# from svy.logging import install_pretty_logging  # the helper we discussed

# install_pretty_logging(logging.INFO)
# logging.getLogger(__name__).addHandler(logging.NullHandler())


def demo():
    raise SvyError(
        title="Dimension mismatch",
        detail="Inputs differ in length/shape.",
        code="DIMENSION_MISMATCH",
        where="svy.mean",
        param="weights",
        expected="len(y)=12",
        got="len=10",
        hint="Align vector lengths and matrix shapes.",
    )


if __name__ == "__main__":
    try:
        demo()
    except SvyError as e:
        # simplest: human-readable one-liner
        print(str(e))
        # pretty ANSI block (colors/emojis) for terminals
        print("\n" + e.ansi())


# # later when catching
# try:
#     demo()
# except SvyError as e:
#     logging.getLogger(__name__).exception(e)  # uses e.__str__ or Rich/ANSI if available
