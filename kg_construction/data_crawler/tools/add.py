from .uri import uri_to_label
import json


def add_entity(info, graph):
    label = uri_to_label(info["@id"])
    code = "MERGE (n:Entity {name: '%s'})" % label.replace("'", "\\'")
    if 'multi-model' in info:
        if 'audio' in info['multi-model']:
            audio = json.dumps(info["multi-model"]["audio"])
            code += " SET n.audio = '%s'" % audio.replace("'", "\\'")
        if 'image' in info['multi-model']:
            image = json.dumps(info["multi-model"]["image"])
            code += " SET n.image = '%s'" % image.replace("'", "\\'")
        if 'visual' in info['multi-model']:
            visual = json.dumps(info["multi-model"]["visual"])
            code += " SET n.visual = '%s'" % visual.replace("'", "\\'")
    graph.run(code)


def add_relation(info, graph):
    label = info["rel"]["label"]
    start = info["start"]["label"]
    end = info["end"]["label"]
    surfaceText = info["surfaceText"]
    if surfaceText:
        surfaceText = surfaceText.replace('"', '\\"')
    weight = info["weight"]
    license = info["license"]
    if license:
        license = license.replace('"', '\\"')

    graph.run('MERGE (:Entity {name: "%s"})' % start.replace('"', '\\"'))
    graph.run('MERGE (:Entity {name: "%s"})' % end.replace('"', '\\"'))

    code = 'MATCH (s: Entity{name: "%s"}), (t:Entity {name: "%s"}) ' \
           'MERGE (s)-[n:%s {surfaceText: "%s", weight: %s, license: "%s"}]->(t)' \
           % (start.replace('"', '\\"'), end.replace('"', '\\"'), label,
              surfaceText, weight, license)
    if 'multi-model' in info:
        if 'audio' in info['multi-model']:
            audio = json.dumps(info["multi-model"]["audio"])
            code += " SET n.audio = '%s'" % audio.replace("'", "\\'")
        if 'image' in info['multi-model']:
            image = json.dumps(info["multi-model"]["image"])
            code += " SET n.image = '%s'" % image.replace("'", "\\'")
        if 'visual' in info['multi-model']:
            visual = json.dumps(info["multi-model"]["visual"])
            code += " SET n.visual = '%s'" % visual.replace("'", "\\'")

    graph.run(code)

