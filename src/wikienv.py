import ast
import json
import time
import gym
import requests
from bs4 import BeautifulSoup

# import wikipedia

# Wikipedia now requires a descriptive User-Agent (policy change ref:
# https://phabricator.wikimedia.org/T400119). Requests without a UA receive
# 403. Use any identifiable string; academic / classroom traffic is fine.
WIKI_HEADERS = {
    "User-Agent": "YonseiNLP-2026-Assignment/1.0 (educational; contact via course staff)",
    "Accept": "text/html",
}
WIKI_TIMEOUT_SEC = 15


def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


class textSpace(gym.spaces.Space):
  def contains(self, x) -> bool:
    """Return boolean specifying if x is a valid member of this space."""
    return isinstance(x, str)


class WikiEnv(gym.Env):

  def __init__(self):
    """
      Initialize the environment.
    """
    super().__init__()
    self.page = None  # current Wikipedia page
    self.obs = None  # current observation
    self.lookup_keyword = None  # current lookup keyword
    self.lookup_list = None  # list of paragraphs containing current lookup keyword
    self.lookup_cnt = None  # current lookup index
    self.steps = 0  # current number of steps
    self.answer = None  # current answer from the agent
    self.observation_space = self.action_space = textSpace()
    self.search_time = 0
    self.num_searches = 0
    
  def _get_obs(self):
    return self.obs

  def _get_info(self):
    return {"steps": self.steps, "answer": self.answer}

  def reset(self, seed=None, return_info=False, options=None):
    # We need the following line to seed self.np_random
    # super().reset(seed=seed)
    self.obs = ("Interact with Wikipedia using search[], lookup[], and "
                "finish[].\n")
    self.page = None
    self.lookup_keyword = None
    self.lookup_list = None
    self.lookup_cnt = None
    self.steps = 0
    self.answer = None
    observation = self._get_obs()
    info = self._get_info()
    return (observation, info) if return_info else observation

  def construct_lookup_list(self, keyword):
    # find all paragraphs
    if self.page is None:
      return []
    paragraphs = self.page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]

    parts = sentences
    parts = [p for p in parts if keyword.lower() in p.lower()]
    return parts

  @staticmethod
  def get_page_obs(page):
    # find all paragraphs
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    # Take more sentences (was 5) so attribute facts like place of death,
    # nationality, etc. are more likely to surface in the lead window.
    return ' '.join(sentences[:10])

  @staticmethod
  def extract_infobox(soup):
    """Extract structured key:value facts from the Wikipedia infobox.

    Returns a short pre-formatted block like:
      [Infobox]
      Born: September 26, 1898 ...
      Died: July 11, 1937 (aged 38), Hollywood, ...
      Nationality: American
      ...

    Returns empty string if no infobox is found.
    """
    infobox = soup.find('table', class_='infobox')
    if not infobox:
      return ""
    rows = []
    for tr in infobox.find_all('tr'):
      th = tr.find('th')
      td = tr.find('td')
      if not th or not td:
        continue
      label = clean_str(th.get_text(separator=" ", strip=True))
      value = clean_str(td.get_text(separator=" ", strip=True))
      # Strip footnote markers and excess whitespace
      value = " ".join(value.split())
      if label and value and len(value) < 200:  # skip overly long cells
        rows.append(f"{label}: {value}")
    if not rows:
      return ""
    return "[Infobox]\n" + "\n".join(rows[:15]) + "\n"


  def search_step(self, entity, _retry_depth=0):
    entity_ = entity.replace(" ", "+")
    search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
    old_time = time.time()
    try:
      resp = requests.get(search_url, headers=WIKI_HEADERS, timeout=WIKI_TIMEOUT_SEC)
    except (requests.RequestException, RecursionError, Exception) as e:
      # Catch everything — broken HTTP responses (e.g., recursion in Python's
      # email header parser) should not kill the run; just skip this query.
      self.search_time += time.time() - old_time
      self.num_searches += 1
      self.obs = f"Wikipedia request failed ({type(e).__name__}) for '{entity}'."
      return
    self.search_time += time.time() - old_time
    self.num_searches += 1
    if resp.status_code != 200:
      self.obs = f"Wikipedia returned HTTP {resp.status_code} for '{entity}'."
      return
    response_text = resp.text
    soup = BeautifulSoup(response_text, features="html.parser")
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
    if result_divs:  # mismatch — disambiguation / listing page
      self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
      # Auto-retry with the top-ranked match (1 level deep) so the model
      # doesn't burn an iteration chasing the suggestion list.
      if _retry_depth < 1 and self.result_titles:
        return self.search_step(self.result_titles[0], _retry_depth=_retry_depth + 1)
      self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
    else:
      page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
      if any("may refer to:" in p for p in page):
        self.search_step("[" + entity + "]")
      else:
        # Prepend structured infobox facts (Born, Died, Nationality, etc.)
        # so attribute-style sub-questions can be answered without needing
        # to scan the body.
        infobox_text = self.extract_infobox(soup)
        self.page = infobox_text
        for p in page:
          if len(p.split(" ")) > 2:
            self.page += clean_str(p)
            if not p.endswith("\n"):
              self.page += "\n"
        self.obs = self.get_page_obs(self.page)
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
  
  def step(self, action):
    reward = 0
    done = False
    action = action.strip()
    if self.answer is not None:  # already finished
      done = True
      return self.obs, reward, done, self._get_info()
    
    if action.startswith("search[") and action.endswith("]"):
      entity = action[len("search["):-1]
      # entity_ = entity.replace(" ", "_")
      # search_url = f"https://en.wikipedia.org/wiki/{entity_}"
      self.search_step(entity)
    elif action.startswith("lookup[") and action.endswith("]"):
      keyword = action[len("lookup["):-1]
      if self.lookup_keyword != keyword:  # reset lookup
        self.lookup_keyword = keyword
        self.lookup_list = self.construct_lookup_list(keyword)
        self.lookup_cnt = 0
      if self.lookup_cnt >= len(self.lookup_list):
        self.obs = "No more results.\n"
      else:
        self.obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
        self.lookup_cnt += 1
    elif action.startswith("finish[") and action.endswith("]"):
      answer = action[len("finish["):-1]
      self.answer = answer
      done = True
      self.obs = f"Episode finished, reward = {reward}\n"
    elif action.startswith("think[") and action.endswith("]"):
      self.obs = "Nice thought."
    else:
      self.obs = "Invalid action: {}".format(action)

    self.steps += 1

    return self.obs, reward, done, self._get_info()
  
  def get_time_info(self):
    speed = self.search_time / self.num_searches if self.num_searches else 0
    return {
        "call_speed": speed,
        "call_time": self.search_time,
        "num_calls": self.num_searches,
    }
