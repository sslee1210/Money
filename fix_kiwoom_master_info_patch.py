from pathlib import Path

ROOT = Path.cwd()
target = ROOT / "kiwoom_bridge_server" / "kiwoom_bridge_flow.py"

if not target.exists():
    raise SystemExit(f"[ERROR] 파일을 찾지 못했습니다: {target}\nMoney 저장소 루트에서 실행하세요. 예: C:\\Users\\sslee\\Desktop\\Money")

text = target.read_text(encoding="utf-8")

# 1) typing import에 List 추가
text = text.replace(
    "from typing import Any, Dict\n",
    "from typing import Any, Dict, List\n",
)

# 2) 기존에 이미 패치되어 있으면 중복 적용 방지
if "def _hydrate_master(self, codes: List[str]) -> None:" not in text:
    marker = "    def _on_receive_real_data(self, code, real_type, real_data) -> None:\n"
    if marker not in text:
        raise SystemExit("[ERROR] 삽입 위치를 찾지 못했습니다. kiwoom_bridge_flow.py 구조가 예상과 다릅니다.")

    method = """    def _hydrate_master(self, codes: List[str]) -> None:
        \""\"Hydrate master data without calling unsupported GetMasterStockInfo.

        Some Kiwoom OpenAPI+ ActiveX installations do not expose
        GetMasterStockInfo(QString). Calling it through QAxWidget repeatedly emits
        noisy QAxBase errors and can make the bridge look broken even when quote/TR
        data is available. The flow bridge only needs name, theme/sector hints, and
        exclusion checks, so it uses GetMasterCodeName plus theme mapping instead.
        \""\"\"

        self._ensure_theme_map()
        for code in codes:
            normalized = base.clean_code(code)
            if normalized in self.master:
                continue
            name = self._code_name(normalized)
            raw_info = ""
            themes = self.theme_by_code.get(normalized, [])
            sector_info = ko.pick_sector(raw_info, name, themes, normalized)
            self.master[normalized] = {
                "code": normalized,
                "name": name,
                "rawInfo": raw_info,
                "sector": sector_info["sector"],
                "sectorSource": sector_info["sectorSource"],
                "themes": sector_info["themes"],
                "excluded": base.is_excluded_name(name),
                "masterInfoSkipped": True,
                "masterInfoSkipReason": "GetMasterStockInfo is not supported by this Kiwoom ActiveX control",
            }

"""
    text = text.replace(marker, method + marker)

target.write_text(text, encoding="utf-8")
print(f"[OK] patched: {target}")
print("[OK] GetMasterStockInfo 우회 패치가 적용되었습니다.")
