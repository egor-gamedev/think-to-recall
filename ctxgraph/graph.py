"""Движок: меню по контексту, раскрытие по выбору модели, рекурсия, сокращения, закрепление.

Генеративная модель здесь не вызывается. Всё, что делает движок, — извлечение из источников,
ранжирование маленьким кодировщиком и учёт того, что модель раскрыла и использовала.
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import replace

import numpy as np

from ctxgraph import text as T
from ctxgraph.embed import Encoder
from ctxgraph.ingest import CODE_EXT, Fragment, ingest
from ctxgraph.menu import Arm, Crossing, Item, Menu, MenuConfig, fit_arm
from ctxgraph.sources.base import Link, Source
from ctxgraph.sources.experience import ExperienceSource
from ctxgraph.store import Edge, Node, Store, diversity

ANCHOR_KINDS = ("keyphrase", "identifier", "lemma")
#: Внешние источники: в техническом контексте (код, SQL, логи) — только при высокой близости.
EXTERNAL = {"wordnet", "ruwordnet", "conceptnet", "wikidata"}
#: Мягкое предпочтение источников при равенстве (WordNet шумнее по разметке раунда 1).
SOURCE_PRIOR = {"wordnet": 0.85}
BAD_POS = {"VERB", "INFN", "ADVB", "NUMR", "PRED", "PREP", "CONJ", "PRCL", "INTJ", "GRND", "NPRO"}


class Engine:
    def __init__(
        self,
        store: Store,
        sources: list[Source],
        encoder: Encoder,
        *,
        config: MenuConfig | None = None,
        session: str | None = None,
        seed: int = 0,
        reinforce_amount: float = 0.5,
    ) -> None:
        self.store = store
        self.sources = sources
        self.encoder = encoder
        self.config = config or MenuConfig()
        self.reinforce_amount = reinforce_amount
        self.rng = random.Random(seed)
        self.new_session(session)

    # ---------- сессия ----------
    def new_session(self, session: str | None = None) -> str:
        self.session = session or uuid.uuid4().hex[:8]
        self._parent: dict[str, str] = {}  # узел → чьё меню его предложило
        self._shown: dict[str, Item] = {}
        self._anchor_root: dict[str, str] = {}  # узел → якорь (ent:...) его рукава
        self._sig: list[str] = []
        self._sig_vec: np.ndarray | None = None
        self._used: set[str] = set()
        for s in self.sources:
            if isinstance(s, ExperienceSource):
                s.current_session = self.session
        return self.session

    def observe(self, context: str | list[dict]) -> list[Fragment]:
        """Записать контекст в опыт текущей сессии (для будущих сессий)."""
        frags = ingest(context)
        for f in frags:
            self.store.put_fragment(f, self.session)
        self.store.commit()
        return frags

    # ---------- вспомогательное ----------
    def _source_for(self, node_id: str) -> Source | None:
        return next((s for s in self.sources if s.owns(node_id)), None)

    def _block_text(self, f: Fragment) -> str:
        if f.text:
            return f.text[:1000]
        # Текст файла не хранится: окружение термина собираем из его сущностей.
        kps = [e.surface for e in f.entities if e.kind in ("keyphrase", "identifier")][:15]
        lems = [e.surface for e in f.entities if e.kind == "lemma"][:25]
        return f"{f.label}. " + ", ".join(kps + lems)

    @staticmethod
    def _is_technical(f: Fragment) -> bool:
        if f.kind in ("code", "tool_result", "tool_call"):
            return True
        if f.kind == "file" and f.label.lower().endswith(CODE_EXT):
            return True
        return sum(e.kind in ("identifier", "code", "path") for e in f.entities) >= 3

    @staticmethod
    def _label_ok(term: str, label: str) -> bool:
        """Название узла должно быть тем же понятием: однословный термин — то же слово,
        многословный — все слова термина и не больше двух лишних."""
        q, lab = term.split(), T.canonical(label).split()
        if not lab:
            return False
        if T.is_cyrillic(term) != T.is_cyrillic(label):
            return True  # совпадение по алиасу на другом языке — доверяем поиску
        if lab == q:
            return True
        return len(q) >= 2 and set(q) <= set(lab) and len(lab) - len(q) <= 2

    def _anchor_candidates(self, f: Fragment, block_vec: np.ndarray | None = None) -> list[str]:
        sess = self.session
        n = max(self.store.n_fragments(sess), 1)
        seen: list[str] = []
        scored: list[tuple[float, str]] = []
        for order, e in enumerate(f.entities):
            if e.kind not in ANCHOR_KINDS or e.canonical in seen:
                continue
            words = e.canonical.split()
            if not words or len(words) > 3 or len(e.canonical) < 3:
                continue
            if T.is_cyrillic(e.canonical):
                tags = [T.pos(w) for w in words]
                if any(t in BAD_POS for t in tags) or "NOUN" not in tags:
                    continue  # якорь — предмет речи: фраза с существительным и без глаголов
            seen.append(e.canonical)
            df = self.store.doc_freq(e.canonical, sess)
            idf = np.log((n + 1) / (df + 1)) + 1
            kind_bonus = {"keyphrase": 1.0, "identifier": 0.9, "lemma": 0.8}[e.kind]
            # Порядок YAKE, редкость, вид сущности и центральность (близость к смыслу блока).
            central = (
                float(self.encoder.one(e.canonical) @ block_vec) if block_vec is not None else 0.5
            )
            score = kind_bonus * idf / (1 + 0.15 * order) * (0.1 + max(central, 0.0)) ** 2
            scored.append((score, e.canonical))
        scored.sort(reverse=True)
        return [c for _, c in scored]

    def _link_term(
        self, term: str, block_vec: np.ndarray, frag: Fragment | None = None
    ) -> list[tuple[Node, float]]:
        """Привязка термина: точное совпадение / лемма → смысл по близости описания к фрагменту.

        Правила (по разметке раунда 1): в техническом фрагменте общие словари — только при
        близости ≥ `tech_dict_min_sim`; узлы-люди, организации, статьи — только если термин
        в контексте написан как имя собственное и совпадает с названием.
        """
        cfg = self.config
        lang = "ru" if T.is_cyrillic(term) else "en"
        technical = frag is not None and self._is_technical(frag)
        surfaces = {e.surface for e in frag.entities if e.canonical == term} if frag else set()
        proper = any(s[:1].isupper() and not s.isupper() for s in surfaces)
        out: list[tuple[Node, float]] = []
        for src in self.sources:
            if not src.available:
                continue
            cands = src.link(term, lang)
            if not cands:
                continue
            best, best_rank, best_s = None, -9.0, -1.0
            for rank, c in enumerate(cands):
                if src.name == "wikidata":
                    if not self._label_ok(term, c.label):
                        continue
                    if c.meta.get("banned") and not (proper and T.canonical(c.label) == term):
                        continue
                gloss = c.meta.get("desc") or c.meta.get("gloss") or ""
                # Ранжируем смыслы по «название + описание», а принимаем по одному описанию:
                # название совпадает с термином у всех смыслов и завышает близость.
                r = float(self.encoder.one(f"{c.label}. {gloss}") @ block_vec)
                r += 0.1 * (T.canonical(c.label) == term) - 0.03 * rank
                if r > best_rank:
                    best, best_rank = c, r
                    best_s = float(self.encoder.one(gloss or c.label) @ block_vec)
            if best is None:
                continue
            need = cfg.link_min_sim
            if technical and src.name in EXTERNAL:
                need = max(need, cfg.tech_dict_min_sim)
            if best_s >= need or src.name == "experience":
                out.append((best, best_s))
        # При равенстве предпочитаем RuWordNet и Викиданные: WordNet шумнее.
        wn = [s for n, s in out if n.source == "wordnet"]
        rivals = [s for n, s in out if n.source in ("ruwordnet", "wikidata")]
        if wn and rivals and max(rivals) >= wn[0] - 0.1:
            out = [(n, s) for n, s in out if n.source != "wordnet"]
        return out

    def _neighbors(self, node: Node) -> list[Link]:
        src = self._source_for(node.id)
        return src.neighbors(node) if src is not None else []

    def _remember(self, parent: str, link: Link) -> None:
        self.store.put_node(link.node)
        self.store.put_edge(
            Edge(
                parent,
                link.node.id,
                link.rel,
                link.source,
                link.slot,
                link.weight,
                provenance=link.via,
            )
        )

    # ---------- сборка рукава ----------
    def _build_arm(
        self,
        *,
        block_id: str,
        block_label: str,
        anchor: str,
        anchor_label: str,
        roots: list[Node],
        block_vec: np.ndarray,
        known: set[str],
        depth: int,
        root_anchor: str,
        shortcut_terms: list[str] | None = None,
        exclude: set[str] | None = None,
    ) -> tuple[Arm, list[tuple[Item, str]]]:
        """Собрать рукав из соседей `roots`. Возвращает рукав и всех кандидатов (для пересечений)."""
        cfg = self.config
        exclude = exclude or set()  # узлы, уже показанные в других рукавах этого меню
        cands: dict[str, tuple[Item, str]] = {}
        for root in roots:
            for link in self._neighbors(root):
                self._remember(root.id, link)
                lab = link.node.label
                canon = T.canonical(lab)
                if not canon or canon in known:
                    continue  # уже есть в контексте — не новость
                e = self.store.edge_between(root.id, link.node.id)
                w = self.store.weight(e) if e else link.weight
                sim = float(self.encoder.one(lab) @ block_vec)
                score = w * (0.4 + 0.6 * max(sim, 0.0)) * SOURCE_PRIOR.get(link.source, 1.0)
                it = Item(link.node.id, lab, link.slot, link.rel, link.source, score, sim, link.via)
                prev = cands.get(canon)
                if prev is None or score > prev[0].score:
                    cands[canon] = (it, root.id)

        # Сокращения: путь, пройденный раньше в похожем контексте, — сразу в «опыт».
        shortcut_items: list[Item] = []
        if depth == 0 and self._sig_vec is not None:
            # Сокращение ищется по всем кандидатам-якорям блока: якорь мог смениться,
            # а путь «рыба → … → лососевые» остаётся полезным в похожем контексте.
            for term in shortcut_terms or [anchor]:
                for sc in self.store.shortcuts_for(f"ent:{term}"):
                    s = float(sc["sig_vec"] @ self._sig_vec)
                    node = self.store.node(sc["target"])
                    if s < cfg.shortcut_sim or node is None or T.canonical(node.label) in known:
                        continue
                    if node.id in exclude or any(i.node_id == node.id for i in shortcut_items):
                        continue
                    via = self.store.node(sc["via"])
                    rel = f"сокращение, через {via.label if via else sc['via']}"
                    shortcut_items.append(
                        Item(
                            node.id,
                            node.label,
                            "experience",
                            rel,
                            "shortcut",
                            1.0 + sc["weight"],
                            s,
                            sc["via"],
                        )
                    )
                    cands.pop(T.canonical(node.label), None)
                    self._parent[node.id] = f"ent:{term}"
            shortcut_items.sort(key=lambda i: -i.score)

        arm = Arm(block_id, block_label, anchor, anchor_label, [r.id for r in roots], depth=depth)
        chosen: set[str] = set()
        by_slot: dict[str, list[Item]] = {}
        for it, _ in cands.values():
            if (
                it.node_id not in exclude
            ):  # уже показан в соседнем рукаве (но остаётся для пересечений)
                by_slot.setdefault(it.slot, []).append(it)
        for slot, lim in cfg.limits.items():
            pool = sorted(by_slot.get(slot, []), key=lambda i: -i.score)
            picked = (shortcut_items if slot == "experience" else []) + pool
            arm.items[slot] = picked[:lim]
            chosen.update(i.node_id for i in arm.items[slot])

        # Далёкое: связано через граф, но мало похоже на блок (квота на новое).
        if cfg.far:
            rest = [
                it
                for it, _ in cands.values()
                if it.node_id not in chosen and it.node_id not in exclude and it.score > 0
            ]
            two_hop = self._two_hop(arm, block_vec, known, chosen | exclude)
            pool = rest + two_hop
            low = [i for i in pool if i.sim < cfg.far_max_sim] or pool
            low.sort(key=lambda i: i.sim)
            if low:
                pick = self.rng.choice(low[:3])
                arm.items["far"] = [replace(pick, slot="far")]
                chosen.add(pick.node_id)

        fit_arm(arm, cfg)
        for it in arm.all_items():
            parent = next((p for i, p in cands.values() if i.node_id == it.node_id), None)
            if it.slot == "far" and it.via:
                parent = it.via
            if it.node_id not in self._parent:
                self._parent[it.node_id] = parent or (roots[0].id if roots else f"ent:{anchor}")
            if parent:
                self.store.count_shown(parent, it.node_id)
            self._shown[it.node_id] = it
            self._anchor_root.setdefault(it.node_id, root_anchor)
        return arm, list(cands.values())

    def _two_hop(
        self, arm: Arm, block_vec: np.ndarray, known: set[str], chosen: set[str]
    ) -> list[Item]:
        """Соседи соседей — только из локальных источников и кэша (без новых сетевых запросов)."""
        out: list[Item] = []
        for slot in ("side", "assoc"):
            for it in arm.items.get(slot, [])[:2]:
                src = self._source_for(it.node_id)
                if src is None:
                    continue
                if src.name in ("wikidata", "conceptnet"):
                    key = f"{src.prefix}nb:{it.node_id.split(':', 1)[1]}"
                    if self.store.cache_get(key) is None:
                        continue
                node = self.store.node(it.node_id)
                if node is None:
                    continue
                for link in src.neighbors(node)[:15]:
                    canon = T.canonical(link.node.label)
                    if link.node.id in chosen or not canon or canon in known:
                        continue
                    self._remember(node.id, link)
                    sim = float(self.encoder.one(link.node.label) @ block_vec)
                    out.append(
                        Item(
                            link.node.id,
                            link.node.label,
                            "far",
                            f"через {it.label}: {link.rel}",
                            link.source,
                            link.weight * 0.5,
                            sim,
                            via=node.id,
                        )
                    )
        return out

    # ---------- публичные инструменты ----------
    def menu(self, context: str | list[dict], *, auto_depth: int | None = None) -> Menu:
        t0 = time.perf_counter()
        cfg = self.config
        depth = cfg.auto_depth if auto_depth is None else auto_depth
        frags = ingest(context)
        known = {e.canonical for f in frags for e in f.entities}
        ctx_text = "\n".join(self._block_text(f) for f in frags)
        self._sig = sorted({c for f in frags for c in f.signature})[:64]
        self._sig_vec = self.encoder.one(ctx_text) if ctx_text else None

        arms: list[Arm] = []
        reach: dict[str, tuple[str, set[str]]] = {}  # node_id → (label, якоря рукавов)
        shown_ids: set[str] = set()
        surface = {e.canonical: e.surface for f in frags for e in f.entities}
        for f in frags[-cfg.max_arms :]:
            if len(arms) >= cfg.max_arms:
                break
            bvec = self.encoder.one(self._block_text(f))
            terms = self._anchor_candidates(f, bvec)
            used_tokens: set[str] = set()
            used_roots: set[str] = set()
            n_block = 0
            for term in terms[: cfg.anchor_tries + cfg.arms_per_block - 1]:
                if n_block >= cfg.arms_per_block or len(arms) >= cfg.max_arms:
                    break
                if used_tokens & set(term.split()):
                    continue  # «аквариум» и «рыба в аквариум» — один и тот же рукав
                linked = [
                    (n, s) for n, s in self._link_term(term, bvec, f) if n.id not in used_roots
                ]
                if not linked:
                    continue
                # Привязка без соседей ничего не даёт меню — такие узлы и рёбра не пишем.
                linked = [(n, s) for n, s in linked if self._neighbors(n)]
                if not linked:
                    continue
                ent = f"ent:{term}"
                self.store.put_node(Node(ent, term, "context"))
                for node, s in linked:
                    self.store.put_node(node)
                    self.store.put_edge(
                        Edge(
                            ent,
                            node.id,
                            "привязка",
                            "link",
                            "link",
                            max(s, 0.0),
                            provenance=f.source,
                        )
                    )
                    self._parent.setdefault(node.id, ent)
                roots = [n for n, _ in linked]
                arm, cands = self._build_arm(
                    block_id=f.id,
                    block_label=f.label,
                    anchor=term,
                    anchor_label=surface.get(term, term),
                    roots=roots,
                    block_vec=bvec,
                    known=known,
                    depth=0,
                    root_anchor=ent,
                    shortcut_terms=terms[:6] if n_block == 0 else [term],
                    exclude=shown_ids,
                )
                if not any(arm.items.values()):
                    continue
                for it, _ in cands:
                    reach.setdefault(it.node_id, (it.label, set()))[1].add(term)
                for r in roots:
                    reach.setdefault(r.id, (r.label, set()))[1].add(term)
                for r in roots:
                    self.store.count_shown(ent, r.id)
                self._auto_expand(arm, depth, bvec, known)
                arms.append(arm)
                shown_ids |= {i.node_id for i in arm.all_items()}
                n_block += 1
                used_tokens |= set(term.split())
                used_roots |= {r.id for r in roots}
        # Пересечение: узел, до которого дотянулись рукава разных якорей, — неочевидная связь.
        crossings = [
            Crossing(nid, lab, sorted(anchors))
            for nid, (lab, anchors) in reach.items()
            if len(anchors) >= 2 and T.canonical(lab) not in known
        ][:3]
        self.store.log("menu", None, self.session, arms=len(arms), auto_depth=depth)
        self.store.commit()
        return Menu(arms, crossings, (time.perf_counter() - t0) * 1000, cfg)

    def _auto_expand(self, arm: Arm, levels: int, bvec: np.ndarray, known: set[str]) -> None:
        """auto_depth: раскрыть лучший пункт ветки без участия модели, `levels` уровней вниз."""
        cur = arm
        for _ in range(levels):
            pool = cur.items.get("deeper") or [
                i for s in ("side", "assoc") for i in cur.items.get(s, [])
            ]
            if not pool:
                return
            best = max(pool, key=lambda i: i.score)
            child = self._expand_arm(best.node_id, bvec, known, depth=cur.depth + 1, auto=True)
            if child is None:
                return
            cur.children.append(child)
            cur = child

    def _expand_arm(
        self,
        node_id: str,
        bvec: np.ndarray | None,
        known: set[str],
        *,
        depth: int,
        auto: bool,
    ) -> Arm | None:
        node = self.store.node(node_id)
        if node is None:
            return None
        parent = self._parent.get(node_id) or self._infer_parent(node_id)
        if parent:
            self.store.count_expansion(parent, node_id)
        self.store.log("auto_expand" if auto else "expand", node_id, self.session)
        vec = (
            bvec
            if bvec is not None
            else (self._sig_vec if self._sig_vec is not None else self.encoder.one(node.label))
        )
        root_anchor = self._anchor_root.get(node_id, parent or node_id)
        arm, _ = self._build_arm(
            block_id=node_id,
            block_label=node.label,
            anchor=node_id,
            anchor_label=node.label,
            roots=[node],
            block_vec=vec,
            known=known | {T.canonical(node.label)},
            depth=depth,
            root_anchor=root_anchor,
        )
        return arm

    def _infer_parent(self, node_id: str) -> str | None:
        """Родитель узла, которого не было в меню: ребро от узла, уже известного этой сессии."""
        known = [
            e
            for e in self.store.edges_to(node_id)
            if e.src in self._parent or e.src.startswith("ent:")
        ]
        if not known:
            return None
        best = max(known, key=self.store.weight)
        self._parent[node_id] = best.src
        return best.src

    def expand(self, node_id: str) -> Arm | None:
        """Меню следующего уровня для выбранного узла."""
        arm = self._expand_arm(node_id, None, set(self._sig), depth=1, auto=False)
        self.store.commit()
        return arm

    def _chain(self, node_id: str) -> list[str]:
        chain = [node_id]
        while chain[-1] in self._parent and len(chain) < 10:
            nxt = self._parent[chain[-1]]
            if nxt in chain:
                break
            chain.append(nxt)
        return chain  # [использованный, ..., якорь ent:...]

    def mark_used(self, node_id: str) -> dict:
        """Связь пригодилась: усилить рёбра пути, при пути ≥ 2 шагов от якоря — сохранить сокращение."""
        if node_id in self._used:
            return {"node": node_id, "reinforced": 0, "shortcut": False}
        self._used.add(node_id)
        if node_id not in self._parent:
            self._infer_parent(node_id)
        chain = self._chain(node_id)
        reinforced = 0
        for child, parent in zip(chain, chain[1:], strict=False):
            reinforced += self.store.reinforce(parent, child, self.reinforce_amount)
        shortcut = False
        anchor = chain[-1] if chain[-1].startswith("ent:") else None
        # chain: [C, B, (L), ent:A]. Сокращение имеет смысл, если C не был прямым соседом привязки.
        hops = [n for n in chain[:-1] if not self._is_link_root(n)]
        if anchor and len(hops) >= 2 and self._sig_vec is not None:
            via = hops[1]
            self.store.add_shortcut(
                anchor, node_id, via, self._sig, self._sig_vec, 1.0, self.session
            )
            shortcut = True
        self.store.log("used", node_id, self.session, chain=chain)
        self.store.commit()
        return {"node": node_id, "reinforced": reinforced, "shortcut": shortcut, "chain": chain}

    def _is_link_root(self, node_id: str) -> bool:
        p = self._parent.get(node_id, "")
        return p.startswith("ent:")

    def reinforce_from_answer(self, answer: str) -> list[str]:
        """Закрепление по пользе: пункт меню попал в итоговый ответ (по леммам или по id)."""
        ans = set(T.content_lemmas(answer))
        used = []
        for nid, it in list(self._shown.items()):
            lem = [w for w in T.content_lemmas(it.label) if len(w) >= 4]
            if nid in answer or (lem and all(w in ans for w in lem)):
                self.mark_used(nid)
                used.append(nid)
        return used

    def diversity(self, last: int | None = None) -> dict:
        nodes = [r[2] for r in self.store.events("expand")]
        return diversity(nodes[-last:] if last else nodes)
