"""Cache immutable external heating tables for one fixed-configuration process."""
def patch_heat_table_cache(text):
    anchor = "int init_heat()\n{"
    assert text.count(anchor) == 1
    text = text.replace(anchor, """
static bool heat_tables_cache_enabled, heat_tables_ready;
static unsigned long long heat_tables_loads, heat_tables_hits;

void SetGPUHeatTableCache(int enabled) {
    heat_tables_cache_enabled = enabled;
    if (!enabled && heat_tables_ready) {
        heat_tables_ready = false;
        destruct_heat();
    }
}

void GetGPUHeatTableCacheStats(unsigned long long *out) {
    out[0] = heat_tables_loads;
    out[1] = heat_tables_hits;
}
""" + anchor + """
    if (heat_tables_cache_enabled && heat_tables_ready) {
        ++heat_tables_hits;
        return 0;
    }
""",1)
    anchor = '    LOG_SUPER_DEBUG("Done initializing heat.");'
    assert text.count(anchor) == 1
    text = text.replace(anchor,anchor+"""
    ++heat_tables_loads;
    if (heat_tables_cache_enabled) heat_tables_ready = true;""",1)
    anchor = "void destruct_heat()\n{"
    assert text.count(anchor) == 1
    return text.replace(anchor,anchor+"\n    if (heat_tables_cache_enabled) return;",1)
