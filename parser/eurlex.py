from datetime import datetime
from http.server import BaseHTTPRequestHandler
import json
from bs4 import BeautifulSoup
import re
from abc import *
import requests


class IJsonifyable(ABC):
    @abstractmethod
    def to_json(self) -> dict:
        pass


class Rule(IJsonifyable, ABC):
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def to_json(self) -> dict:
        return {'name': self.name, 'description': self.description}


class SplitRule(Rule):
    def __init__(self, name, description):
        super().__init__(name, description)
        self.translations = {"part": ["part", "deel"], "title": ["title", "titel"], "chapter": [
            "chapter", "hoofdstuk"], "section": ["section", "afdeling"], "subsection": ["subsection", "subafdeling"], "article": ["article", "artikel"]}

    def check_fragment_type(self, s: str, name: str) -> bool:
        for w in self.translations[name]:
            if len(re.findall("\W*{}\s\w+\n*".format(w), s.lower())):
                return True
        return False

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        pass


class TransformRule(Rule):
    @abstractmethod
    def transform_text(self, text: str) -> str:
        pass


class TemplateInfo(IJsonifyable):
    def __init__(self, name: str = '', description: str = '', creator: str = '',  date: datetime = datetime.now()) -> None:
        self.name = name
        self.description = description
        self.creator = creator
        self.date = date

    def to_json(self) -> dict:
        return {'name': self.name, 'description': self.description, 'creator': self.creator, 'date': str(self.date)}


class Template(IJsonifyable):
    def __init__(self, split_rule: SplitRule, transform_rules: list[TransformRule], info: TemplateInfo = TemplateInfo()) -> None:
        self.split_rule = split_rule
        self.transform_rules = transform_rules
        self.info = info

    def to_json(self) -> dict:
        return {'info': self.info.to_json(), 'split_rule': self.split_rule.to_json(), 'transform_rules': [rule.to_json() for rule in self.transform_rules]}

    def process_text(self, text: str) -> list[str]:
        texts = self.split_rule.split_text(text)
        res = []
        for t in texts:
            for r in self.transform_rules:
                t = r.transform_text(t)
            res.append(t)
        return res


class SplitHTMLArticle(SplitRule):

    def split_text(self, text: str) -> list[str]:
        res = []
        soup = BeautifulSoup(text, 'html.parser')
        oj_format = False

        def get_field(s):
            if oj_format:
                return 'oj-' + s
            return s

        # introduction
        res.append("")
        current = soup.find(attrs={"class": "doc-ti"})
        if current == None:
            oj_format = True
            current = soup.find(attrs={"class": "oj-doc-ti"})
        while current != None:
            try:
                if current.name != 'p':
                    break
            except:
                pass
            res[-1] += current.get_text() + '\n\n'
            current = current.find_next_sibling()
        #print(res)

        # recitals
        res.append("")
        while current != None:
            try:
                if current.name != 'table':
                    if not (current.name == 'p' and get_field('normal') in current['class']):
                        break
            except:
                pass
            res[-1] += current.get_text() + '\n\n'
            current = current.find_next_sibling()

        # body
        self.total_articles = 0

        def get_articles(header, art):
            self.total_articles += 1
            res.append(header)
            current = art
            res[-1] += current.get_text() + '\n\n'
            current = current.find_next_sibling()
            while current != None:
                try:
                    if get_field('ti-art') in current['class']:
                        self.total_articles += 1
                        res.append("\n\n".join(to_write))
                    if get_field('final') in current['class'] or get_field('ti-section-1') in current['class']:
                        break
                except:
                    pass
                res[-1] += current.get_text() + '\n\n'
                if current.find_next_sibling() == None:
                    potential_art = current.parent.find_next_sibling().find(
                        attrs={'class': get_field('ti-art')})
                    if potential_art != None:
                        current = potential_art
                    else:
                        break
                else:
                    current = current.find_next_sibling()

        ti_secs = soup.find_all(attrs={"class": get_field("ti-section-1")})
        to_write = []
        level = 6
        if len(ti_secs) == 0:
            get_articles("", soup.find(
                attrs={"class": get_field("ti-art")}))
        else:
            for ti in ti_secs:
                s = ti.get_text()
                if self.check_fragment_type(s, "part"):
                    to_write = []
                    to_write.append(
                        s + '\n\n' + ti.find_next_sibling().get_text() + '\n\n')
                    level = 5
                elif self.check_fragment_type(s, "title"):
                    if level < 5:
                        to_write = to_write[:-(4-level+1)]
                    to_write.append(
                        s + '\n\n' + ti.find_next_sibling().get_text() + '\n\n')
                    level = 4
                elif self.check_fragment_type(s, "chapter"):
                    if level < 4:
                        to_write = to_write[:-(3-level+1)]
                    to_write.append(
                        s + '\n\n' + ti.find_next_sibling().get_text() + '\n\n')
                    level = 3
                elif self.check_fragment_type(s, "section"):
                    if level < 3:
                        to_write = to_write[:-(2-level+1)]
                    to_write.append(
                        s + '\n\n' + ti.find_next_sibling().get_text() + '\n\n')
                    level = 2
                elif self.check_fragment_type(s, "subsection"):
                    if level < 2:
                        to_write = to_write[:-(1-level+1)]
                    to_write.append(
                        s + '\n\n' + ti.find_next_sibling().get_text() + '\n\n')
                    level = 1
                try:
                    potential_art = ti.find_next_sibling().find_next_sibling()
                    if get_field('ti-art') in potential_art['class']:
                        get_articles("\n\n".join(to_write), potential_art)
                    else:
                        potential_art = potential_art.find(
                            attrs={'class': get_field('ti-art')})
                        if potential_art != None:
                            get_articles("\n\n".join(to_write), potential_art)
                except:
                    potential_art = potential_art.find(
                        attrs={'class': get_field('ti-art')})
                    if potential_art != None:
                        get_articles("\n\n".join(to_write), potential_art)
                    pass

        # notes (may want to do it like articles just in case)
        res.append("")
        current = soup.find(
            'hr', attrs={"class": get_field("note")}).find_next_sibling()
        while current != None:
            try:
                if get_field('doc-sep') in current['class'] or get_field('doc-end') in current['class']:
                    break
            except:
                pass
            res[-1] += current.get_text() + '\n\n'
            current = current.find_next_sibling()

        # annexes
        docseps = soup.find_all(attrs={'class': get_field('doc-sep')})
        self.total_annexes = len(docseps)
        for sep in docseps:
            res.append("")
            res[-1] += sep.find_next_sibling().get_text()

        # trim unnecesary newlines from soup
        def trim_new_lines(s):
            def merge_lines(match_obj):
                if match_obj.group() is not None:
                    return re.sub(r'(\n+)$', '\t', match_obj.group())

            trimmed = re.sub(r'\n+', '\n\n', s)
            trimmed = re.sub(r'\n+(\(?\w+\)|\—|\w+\.)\n+',
                             merge_lines, trimmed)
            c = 0
            for i in range(len(trimmed)):
                if trimmed[i] == '\n':
                    c += 1
                else:
                    break
            return trimmed[c:]

        return list(map(trim_new_lines, res))


def get_text_from(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.text


def process_text(text):
    ti = TemplateInfo(
        'Legislation', 'This template is more suitable for preprocessing European Legislations', 'Carlos Aguilera')

    split_rule = SplitHTMLArticle(
        'Split html into articles', 'Split html files from https://eur-lex.europa.eu/ into articles')
    transform_rules = []
    t = Template(split_rule, transform_rules, ti)
    pre = t.process_text(text)
    res = {}
    res['order'] = ['introduction', 'recitals', 'articles', 'notes', 'annexes']
    res['introduction'] = [pre[0]]
    res['recitals'] = [pre[1]]
    res['articles'] = pre[2:t.split_rule.total_articles+2]
    res['notes'] = [pre[t.split_rule.total_articles+2]]
    res['annexes'] = pre[t.split_rule.total_articles+3:len(pre)]
    return res


class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        content_len = int(self.headers.get('content-length', 0))
        url = self.rfile.read(content_len).decode()
        text = get_text_from(url)
        res = process_text(text)
        self.wfile.write(json.dumps(res).encode())
        return
