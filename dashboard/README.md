# Money Dashboard

Money 안에 포함된 MoneyBoard 대시보드입니다.

## Kiwoom bridge prerequisites

Kiwoom OpenAPI+ is a 32-bit ActiveX/COM control, so the bridge must run with
32-bit Python. A normal 64-bit Python install is not enough.

Install on each Windows PC that will run the Kiwoom bridge:

1. Kiwoom OpenAPI+ and the Kiwoom trading app login environment.
2. Python Windows installer, 32-bit / x86. Python 3.10 is preferred, but any
   supported 32-bit Python detected by `py -0p` can be used.
3. During Python setup, enable the Python Launcher option.

Official Python 3.10.11 download page:
https://www.python.org/downloads/release/python-31011/

Check the installed Python runtimes:

```bat
py -0p
py -3.10-32 -c "import struct; print(struct.calcsize('P') * 8)"
```

If only another 32-bit version is installed, use that selector instead:

```bat
py -3.13-32 -c "import struct; print(struct.calcsize('P') * 8)"
```

The command must print `32`.

Money 통합 실행 파일을 쓰는 방법:

```bat
..\Start_Money_All.bat
```

대시보드만 따로 실행하는 방법:

```bash
npm install
npm run server
```

대시보드는 자체 브릿지를 실행하지 않고 `KIWOOM_BRIDGE_URL`의 Money 브릿지를 사용합니다.
기본 주소는 `http://127.0.0.1:8765`입니다.
