"""Bound report repair within the current Episode without consuming a version."""


class ProtocolPaused(RuntimeError):
    pass


def check_protocol_paused(active):
    if active.get("phase") == "paused_protocol":
        raise ProtocolPaused(active.get("protocol_error", "Episode report recovery exhausted"))


def pause_protocol(active, save, reason):
    active.update(phase="paused_protocol", protocol_error=reason)
    save(active)
    raise ProtocolPaused(reason)
