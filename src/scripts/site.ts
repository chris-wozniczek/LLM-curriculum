const $ = <T extends HTMLElement>(s: string) => document.querySelector<T>(s);

// scroll progress
const bar = $("#progress");
const onScroll = () => {
  const h = document.documentElement;
  const p = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight);
  if (bar) bar.style.transform = `scaleX(${p})`;
};
addEventListener("scroll", onScroll, { passive: true });
onScroll();

// reveal
const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { rootMargin: "0px 0px -8% 0px" });
document.querySelectorAll<HTMLElement>(".grid").forEach((grid) => {
  grid.querySelectorAll<HTMLElement>(".card").forEach((card, index) => {
    card.classList.add("rv");
    card.style.setProperty("--d", `${index * 55}ms`);
  });
});
document.querySelectorAll(".rv").forEach((el) => io.observe(el));

// glossary tooltips
const tips = (window as any).__tips as Record<string, [string, string]>;
const tip = $("#tip")!;
document.addEventListener("mouseover", (e) => {
  const a = (e.target as HTMLElement).closest<HTMLElement>(".term");
  if (!a) return;
  const t = tips[a.dataset.k!];
  if (!t) return;
  tip.innerHTML = `<b></b><span></span>`;
  tip.querySelector("b")!.textContent = t[0];
  tip.querySelector("span")!.textContent = t[1];
  const r = a.getBoundingClientRect();
  const x = Math.min(Math.max(8, r.left + scrollX), scrollX + innerWidth - 316);
  tip.style.left = `${x}px`;
  tip.style.top = `${r.bottom + scrollY + 8}px`;
  tip.classList.add("show");
});
document.addEventListener("mouseout", (e) => { if ((e.target as HTMLElement).closest(".term")) tip.classList.remove("show"); });

// toc highlight
const links = [...document.querySelectorAll<HTMLAnchorElement>(".toc a[href^='#']")];
if (links.length) {
  const map = new Map(links.map((l) => [decodeURIComponent(l.hash.slice(1)), l]));
  const tio = new IntersectionObserver((es) => {
    es.forEach((e) => {
      if (e.isIntersecting) { links.forEach((l) => l.classList.remove("on")); map.get(e.target.id)?.classList.add("on"); }
    });
  }, { rootMargin: "-20% 0px -70% 0px" });
  map.forEach((_, id) => { const h = document.getElementById(id); if (h) tio.observe(h); });
}

// progress tracking
const KEY = "llmc-done";
const done = new Set<string>(JSON.parse(localStorage.getItem(KEY) || "[]"));
const paint = () => {
  document.querySelectorAll<HTMLElement>("[data-done]").forEach((el) => el.classList.toggle("on", done.has(el.dataset.done!)));
  const c = $("#done-count");
  if (c) c.textContent = String(done.size);
};
document.addEventListener("click", (e) => {
  const b = (e.target as HTMLElement).closest<HTMLElement>("button[data-done]");
  if (!b) return;
  const k = b.dataset.done!;
  done.has(k) ? done.delete(k) : done.add(k);
  localStorage.setItem(KEY, JSON.stringify([...done]));
  paint();
});
paint();

// search (pagefind, loaded on first open)
const dlg = $("#search-dlg")!, inp = $<HTMLInputElement>("#search-in")!, out = $("#search-out")!;
let pf: any = null, sel = 0, seq = 0;
const openSearch = async () => {
  dlg.classList.add("open"); inp.focus(); inp.select();
  if (!pf) {
    try { const url = "/pagefind/pagefind.js"; pf = await import(/* @vite-ignore */ url); await pf.init(); if (inp.value.trim()) inp.dispatchEvent(new Event("input")); }
    catch { out.innerHTML = `<div class="search-empty">Search index is built on <code>npm run build</code>. Run a build to use search locally.</div>`; }
  }
};
const closeSearch = () => dlg.classList.remove("open");
$("#search-open")!.addEventListener("click", openSearch);
dlg.addEventListener("click", (e) => { if (e.target === dlg) closeSearch(); });
addEventListener("keydown", (e) => {
  const typing = /INPUT|TEXTAREA/.test((e.target as HTMLElement).tagName);
  if ((e.key === "/" && !typing) || (e.key === "k" && (e.metaKey || e.ctrlKey))) { e.preventDefault(); openSearch(); }
  if (!dlg.classList.contains("open")) return;
  const items = [...out.querySelectorAll("a")];
  if (e.key === "Escape") closeSearch();
  if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); sel = (sel + (e.key === "ArrowDown" ? 1 : -1) + items.length) % Math.max(1, items.length); items.forEach((a, i) => a.classList.toggle("sel", i === sel)); items[sel]?.scrollIntoView({ block: "nearest" }); }
  if (e.key === "Enter" && items[sel]) { location.href = items[sel].href; closeSearch(); }
});
inp.addEventListener("input", async () => {
  if (!pf) return;
  const my = ++seq, q = inp.value.trim();
  if (!q) { out.innerHTML = `<div class="search-empty">Type to search every lesson and the glossary.</div>`; return; }
  const r = await pf.debouncedSearch(q, {}, 120);
  if (!r || my !== seq) return;
  const data = await Promise.all(r.results.slice(0, 8).map((x: any) => x.data()));
  if (my !== seq) return;
  const rows: string[] = [];
  data.forEach((d: any) => {
    const subs = (d.sub_results || []).filter((s: any) => s.url !== d.url).slice(0, 2);
    rows.push(`<a href="${d.url}"><small>${d.meta?.title || ""}</small><b>${d.meta?.title || d.url}</b><p>${d.excerpt}</p></a>`);
    subs.forEach((s: any) => rows.push(`<a href="${s.url}"><small>${d.meta?.title || ""} ›</small><b>${s.title}</b><p>${s.excerpt}</p></a>`));
  });
  sel = 0;
  out.innerHTML = rows.length ? rows.join("") : `<div class="search-empty">Nothing found for “${q.replace(/</g, "&lt;")}”.</div>`;
  out.querySelector("a")?.classList.add("sel");
});
