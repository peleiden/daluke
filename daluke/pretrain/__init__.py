from pelutils import TickTock


def close_tt(tt: TickTock):
    while True:
        try:
            tt.end_profile()
        except IndexError:
            break
