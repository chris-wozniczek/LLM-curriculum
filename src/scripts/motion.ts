const root = document.documentElement;
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

const curtain = document.getElementById("curtain");
if (root.classList.contains("intro") && curtain) {
  const fallback = (window as Window & { __llmcIntroFallback?: number }).__llmcIntroFallback;
  if (fallback) clearTimeout(fallback);
  const counter = document.getElementById("curtain-count");
  const started = performance.now();
  const count = (now: number): void => {
    const progress = Math.min(1, (now - started) / 1100);
    if (counter) counter.textContent = String(Math.round((1 - Math.pow(1 - progress, 3)) * 100)).padStart(3, "0");
    if (progress < 1) requestAnimationFrame(count);
  };
  requestAnimationFrame(count);
  setTimeout(() => {
    curtain.classList.add("lift");
    root.classList.add("ready");
    setTimeout(() => {
      root.classList.remove("intro");
      curtain.remove();
      window.dispatchEvent(new Event("llmc:ready"));
    }, 1150);
  }, 1400);
}

if (!reduce) {
  const splitText = (el: HTMLElement): void => {
    if (el.dataset.split === "1") return;
    el.dataset.split = "1";
    let i = 0;
    const walk = (node: Node): void => {
      for (const child of Array.from(node.childNodes)) {
        if (child.nodeType === Node.TEXT_NODE) {
          const pieces = (child.textContent ?? "").match(/\s+|\S+/g) ?? [];
          const fragment = document.createDocumentFragment();
          for (const piece of pieces) {
            if (/^\s+$/.test(piece)) {
              fragment.append(document.createTextNode(piece));
              continue;
            }
            const word = document.createElement("span");
            word.className = "w";
            const text = document.createElement("span");
            text.style.setProperty("--i", String(i++));
            text.textContent = piece;
            word.append(text);
            fragment.append(word);
          }
          child.replaceWith(fragment);
        } else if (child.nodeType === Node.ELEMENT_NODE && (child as HTMLElement).tagName !== "BR") {
          walk(child);
        }
      }
    };
    walk(el);
    el.classList.add("split");
  };

  const headings = document.querySelectorAll<HTMLElement>(".display, .lesson-head h1, .prose h2");
  headings.forEach(splitText);
  const headingObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("in");
      headingObserver.unobserve(entry.target);
    });
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
  headings.forEach((heading) => headingObserver.observe(heading));

  const topbar = document.querySelector<HTMLElement>(".topbar");
  const search = document.getElementById("search-dlg");
  let lastY = scrollY;
  addEventListener("scroll", () => {
    const y = scrollY;
    if (search?.classList.contains("open")) topbar?.classList.remove("hide");
    else if (topbar) topbar.classList.toggle("hide", y > 240 && y > lastY + 2);
    if (y < lastY - 2) topbar?.classList.remove("hide");
    lastY = y;
  }, { passive: true });
  if (topbar && search) {
    new MutationObserver(() => {
      if (search.classList.contains("open")) topbar.classList.remove("hide");
    }).observe(search, { attributes: true, attributeFilter: ["class"] });
  }

  const fine = matchMedia("(hover: hover) and (pointer: fine)").matches;
  const cursor = document.getElementById("cursor");
  if (fine && cursor) {
    let x = innerWidth / 2;
    let y = innerHeight / 2;
    let cx = x;
    let cy = y;
    let frame = 0;
    const dot = document.createElement("i");
    dot.className = "cursor-dot";
    dot.setAttribute("aria-hidden", "true");
    document.body.append(dot);
    const move = (event: PointerEvent): void => {
      x = event.clientX;
      y = event.clientY;
      dot.style.transform = `translate(${x}px, ${y}px)`;
      if (!frame) frame = requestAnimationFrame(tick);
    };
    const tick = (): void => {
      cx += (x - cx) * 0.16;
      cy += (y - cy) * 0.16;
      cursor.style.transform = `translate(${cx}px, ${cy}px)`;
      if (Math.abs(x - cx) + Math.abs(y - cy) > 0.15) frame = requestAnimationFrame(tick);
      else {
        cx = x;
        cy = y;
        cursor.style.transform = `translate(${cx}px, ${cy}px)`;
        frame = 0;
      }
    };
    addEventListener("pointermove", move, { passive: true });
    document.addEventListener("pointerover", (event) => {
      const target = (event.target as HTMLElement).closest("a, button, [role='button'], input, label, summary");
      cursor.classList.toggle("big", Boolean(target));
    });
  } else cursor?.remove();

  document.querySelectorAll<HTMLElement>(".card").forEach((card) => {
    card.addEventListener("pointermove", (event) => {
      const rect = card.getBoundingClientRect();
      card.style.setProperty("--mx", `${((event.clientX - rect.left) / rect.width) * 100}%`);
      card.style.setProperty("--my", `${((event.clientY - rect.top) / rect.height) * 100}%`);
    }, { passive: true });
  });

  document.querySelectorAll<HTMLElement>(".fig").forEach((figure) => {
    const values = new Map<HTMLElement, string>();
    figure.querySelectorAll<HTMLElement>("[data-num]").forEach((value) => values.set(value, value.textContent ?? ""));
    const timers = new Map<Element, number>();
    const bump = (value: Element): void => {
      value.classList.add("bump");
      const oldTimer = timers.get(value);
      if (oldTimer) clearTimeout(oldTimer);
      timers.set(value, window.setTimeout(() => {
        value.classList.remove("bump");
        timers.delete(value);
      }, 500));
    };
    const collectElements = (nodes: NodeList): Element[] | null => {
      const elements: Element[] = [];
      const visit = (node: Node): boolean => {
        if (!(node instanceof Element)) return true;
        elements.push(node);
        if (elements.length > 400) return false;
        for (const child of Array.from(node.children)) {
          if (!visit(child)) return false;
        }
        return true;
      };
      for (const node of Array.from(nodes)) {
        if (!visit(node)) return null;
      }
      return elements;
    };
    const animatePair = (oldEl: Element, newEl: Element): void => {
      const oldText = oldEl.textContent ?? "";
      const newText = newEl.textContent ?? "";
      if (oldEl.children.length === 0 && newEl.children.length === 0 && oldText !== newText && /\d/.test(newText)) bump(newEl);

      const oldStyle = document.createElement("span").style;
      const newStyle = document.createElement("span").style;
      oldStyle.cssText = oldEl.getAttribute("style") ?? "";
      newStyle.cssText = newEl.getAttribute("style") ?? "";
      const styleFrom: Keyframe = {};
      const styleTo: Keyframe = {};
      const styleProperties = ["width", "height", "left", "top", "flex-basis", "transform"];
      const defaults: Record<string, string> = { width: "auto", height: "auto", left: "auto", top: "auto", "flex-basis": "auto", transform: "none" };
      styleProperties.forEach((property) => {
        const oldValue = oldStyle.getPropertyValue(property);
        const newValue = newStyle.getPropertyValue(property);
        if (oldValue === newValue) return;
        const key = property.replace(/-([a-z])/g, (_, letter: string) => letter.toUpperCase());
        styleFrom[key] = oldValue || defaults[property];
        styleTo[key] = newValue || defaults[property];
      });
      const options = { duration: 320, easing: "cubic-bezier(.22,1,.36,1)" };
      if (Object.keys(styleFrom).length) newEl.animate([styleFrom, styleTo], options);

      const tag = oldEl.tagName.toLowerCase();
      if (!["rect", "circle", "line"].includes(tag)) return;
      const geometryFrom: Keyframe = {};
      const geometryTo: Keyframe = {};
      ["x", "y", "width", "height", "cx", "cy", "r", "x1", "x2", "y1", "y2"].forEach((property) => {
        const oldValue = oldEl.getAttribute(property);
        const newValue = newEl.getAttribute(property);
        if (oldValue === null || newValue === null) return;
        const oldNumber = Number(oldValue);
        const newNumber = Number(newValue);
        if (!Number.isFinite(oldNumber) || !Number.isFinite(newNumber) || oldNumber === newNumber) return;
        geometryFrom[property] = `${oldNumber}px`;
        geometryTo[property] = `${newNumber}px`;
      });
      if (Object.keys(geometryFrom).length) {
        try {
          newEl.animate([geometryFrom, geometryTo], options);
        } catch {}
      }
    };
    const observer = new MutationObserver((records) => {
      const changed = new Set<HTMLElement>();
      for (const record of records) {
        if (record.removedNodes.length && record.addedNodes.length) {
          const oldElements = collectElements(record.removedNodes);
          const newElements = collectElements(record.addedNodes);
          if (!oldElements || !newElements || oldElements.length !== newElements.length) continue;
          if (oldElements.some((oldEl, index) => {
            const newEl = newElements[index];
            const oldClass = oldEl.getAttribute("class") ?? "";
            const newClass = newEl.getAttribute("class") ?? "";
            return oldEl.tagName !== newEl.tagName || oldClass !== newClass;
          })) continue;
          oldElements.forEach((oldEl, index) => animatePair(oldEl, newElements[index]));
        }
        const target = record.target instanceof Element ? record.target : record.target.parentElement;
        const value = target?.closest<HTMLElement>("[data-num]");
        if (value) changed.add(value);
      }
      changed.forEach((value) => {
        const text = value.textContent ?? "";
        if (values.get(value) !== text) {
          values.set(value, text);
          bump(value);
        }
      });
    });
    observer.observe(figure, { subtree: true, childList: true, characterData: true });
  });
}
