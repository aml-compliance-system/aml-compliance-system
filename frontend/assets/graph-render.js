/* Lightweight SVG graph renderer. No external graph library is required. */

const NS = "http://www.w3.org/2000/svg";

export function renderGraph(container, data, handlers = {}) {
  const width = 1100;
  const height = 680;
  container.innerHTML = "";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Transaction relationship graph");
  const defs = document.createElementNS(NS, "defs");
  defs.innerHTML = `<marker id="arrow" viewBox="0 0 10 10" refX="18" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#5f8ca3"/></marker>`;
  svg.appendChild(defs);
  const viewport = document.createElementNS(NS, "g");
  svg.appendChild(viewport);
  container.appendChild(svg);

  const nodes = data.nodes || [];
  const edges = data.edges || [];
  const byId = new Map(nodes.map((node) => [node.id, { ...node }]));
  const focal = nodes.find((node) => node.is_focal) || nodes[0];
  const centerX = width / 2;
  const centerY = height / 2;

  // Group accounts by shortest hop, then lay each ring out radially.
  const rings = new Map();
  nodes.forEach((node) => {
    if (!byId.has(node.id)) return;
    const key = node.is_focal ? 0 : Math.max(1, Number(node.hop || 1));
    if (!rings.has(key)) rings.set(key, []);
    rings.get(key).push(node);
  });
  const ringKeys = [...rings.keys()].sort((a, b) => a - b);
  ringKeys.forEach((hop, ringIndex) => {
    const group = rings.get(hop).filter((node) => node.type === "account");
    const holders = rings.get(hop).filter((node) => node.type === "holder");
    const count = group.length;
    const radius = hop === 0 ? 0 : 200 + (ringIndex - 1) * 140;
    group.forEach((node, index) => {
      const angle = (Math.PI * 2 * index) / Math.max(1, count) - Math.PI / 2;
      const entry = byId.get(node.id);
      entry.x = centerX + Math.cos(angle) * radius;
      entry.y = centerY + Math.sin(angle) * radius;
    });
    holders.forEach((node, index) => {
      const angle = (Math.PI * 2 * index) / Math.max(1, holders.length) - Math.PI / 2;
      const entry = byId.get(node.id);
      entry.x = centerX + Math.cos(angle) * (radius + 38);
      entry.y = centerY + Math.sin(angle) * (radius + 38);
    });
  });

  const edgeLayer = document.createElementNS(NS, "g");
  const nodeLayer = document.createElementNS(NS, "g");
  viewport.appendChild(edgeLayer);
  viewport.appendChild(nodeLayer);

  const edgeElements = new Map();
  edges.forEach((edge) => {
    const source = byId.get(edge.source);
    const target = byId.get(edge.target);
    if (!source || !target || source.x === undefined || target.x === undefined) return;
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", source.x);
    line.setAttribute("y1", source.y);
    line.setAttribute("x2", target.x);
    line.setAttribute("y2", target.y);
    line.setAttribute("class", "edge-line");
    line.setAttribute("marker-end", "url(#arrow)");
    if (edge.type === "holder") line.setAttribute("stroke-dasharray", "3 3");
    line.dataset.ref = edge.transaction_ref || edge.id;
    line.dataset.edgeId = edge.id;
    const title = document.createElementNS(NS, "title");
    title.textContent = edge.type === "holder"
      ? `Holder link: ${edge.source} → ${edge.target}`
      : `${edge.transaction_ref || edge.id} · ${edge.timestamp || ""} · ${edge.amount ?? ""} ${edge.currency || ""}`;
    line.appendChild(title);
    line.addEventListener("click", (event) => {
      event.stopPropagation();
      handlers.onEdge?.(edge);
    });
    line.style.cursor = "pointer";
    edgeLayer.appendChild(line);
    edgeElements.set(edge.id, line);
  });

  const nodeElements = new Map();
  nodes.forEach((node) => {
    const entry = byId.get(node.id);
    if (!entry || entry.x === undefined) return;
    const group = document.createElementNS(NS, "g");
    group.setAttribute("transform", `translate(${entry.x}, ${entry.y})`);
    group.style.cursor = "pointer";
    if (node.type === "holder") {
      const rect = document.createElementNS(NS, "rect");
      rect.setAttribute("x", -7);
      rect.setAttribute("y", -7);
      rect.setAttribute("width", 14);
      rect.setAttribute("height", 14);
      rect.setAttribute("rx", 3);
      rect.setAttribute("class", "node-holder");
      group.appendChild(rect);
    } else {
      const circle = document.createElementNS(NS, "circle");
      circle.setAttribute("r", node.is_focal ? 14 : node.hop >= 2 ? 7 : 9);
      circle.setAttribute("fill", node.is_focal ? "#14b8a6" : node.hop >= 2 ? "#64748b" : "#38bdf8");
      circle.setAttribute("class", `node-circle ${node.is_focal ? "focal" : ""}`);
      group.appendChild(circle);
    }
    const label = document.createElementNS(NS, "text");
    label.setAttribute("x", 0);
    label.setAttribute("y", node.is_focal ? 26 : 18);
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("class", "node-label");
    label.textContent = node.type === "holder" ? (node.label || "holder").slice(0, 26) : node.label;
    group.appendChild(label);
    const title = document.createElementNS(NS, "title");
    title.textContent = `${node.label}${node.holder_name ? ` · ${node.holder_name}` : ""}${node.is_focal ? " (focus account)" : ""}`;
    group.appendChild(title);
    group.addEventListener("click", (event) => {
      event.stopPropagation();
      handlers.onNode?.(node);
    });
    group.addEventListener("dblclick", (event) => {
      event.stopPropagation();
      handlers.onExpand?.(node);
    });
    nodeLayer.appendChild(group);
    nodeElements.set(node.id, group);
  });

  let scale = 1;
  const applyScale = () => {
    viewport.setAttribute("transform", `translate(${centerX} ${centerY}) scale(${scale}) translate(${-centerX} ${-centerY})`);
  };
  applyScale();
  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    scale = Math.max(0.45, Math.min(2.5, scale * (event.deltaY > 0 ? 0.9 : 1.1)));
    applyScale();
  }, { passive: false });

  function highlight({ refs = [], nodeIds = [], finding = null } = {}) {
    const refSet = new Set(refs);
    const nodeSet = new Set(nodeIds);
    edgeElements.forEach((line) => {
      const active = refSet.size === 0 || refSet.has(line.dataset.ref);
      line.classList.toggle("highlight", Boolean(finding) && active);
      line.style.opacity = finding && !active ? "0.16" : "";
    });
    nodeElements.forEach((element, id) => {
      const active = nodeSet.size === 0 || nodeSet.has(id);
      element.style.opacity = finding && !active ? "0.35" : "";
      element.querySelector(".node-circle")?.classList.toggle("highlight", Boolean(finding) && active);
    });
  }

  return {
    zoomIn: () => { scale = Math.min(2.5, scale * 1.2); applyScale(); },
    zoomOut: () => { scale = Math.max(0.45, scale / 1.2); applyScale(); },
    reset: () => { scale = 1; applyScale(); highlight(); },
    highlight,
  };
}
