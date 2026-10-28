# Queued behind the running step, so a client stays connected throughout:
# wait until the laptop has uploaded the item files (READY last), then finish.
def wait_then_second_half(timeout=2400):
    need = [f"{WORK}/data/mc_items/{n}" for n in ("own_sample.jsonl.gz", "ptrue_candidates.jsonl.gz", "READY")]
    t0 = time.time()
    while not all(os.path.exists(p) for p in need):
        if time.time() - t0 > timeout:
            log("gave up waiting for the item files")
            return
        print(time.strftime("%H:%M:%S"), "waiting for item files", flush=True)
        time.sleep(30)
    log("item files present; starting the second half")
    second_half()


wait_then_second_half()
