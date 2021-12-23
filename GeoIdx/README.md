# Geopolitical Index

## Geopolitical Threats

```regex
(([G,g]eopoli(tical)*).*(risk|concern|tension|uncertaint))|(^(?=.*(United States))(?=.*tension)(?=.*(military|war|geopolitical|coup|guerrilla|warfare))(?=.*((Latin America)|(Central America)|(South America)|Europe|Africa|(Middle East)|(Far East)|Asia)).*$)
```

## Nuclear Threats

```
^(?=.*((tomic war)|(uclear war)|(uclear conflict)|(tomic conflict)|(uclear missile)))(?=.*(fear|threat|risk|peril|menace)).*$
```

## War Threats

```
(war risk)|(risk.* of war)|(fear of war)|(war fear)|(military threat)|(war threat)|(threat of war)|(^(?=.*((military action)|(military operation)|(military force)))(?=.*(risk|threat)).*$)
```

## Terrorist Threats

```
(errorist threat)|(hreat of terrorism)|(errorism menace)|(enace of terrorism)|(errorist risk)|(error risk)|(isk of [T,t]errorism)|(error threat)
```

## War Acts

```
(eginning of the war)|(utbreak of the war)|(nset of the war)|(scalation of the war)|(tart of the war)|(^(?=.*(war|militar))(?=.*(ir strike)).*$)|(^(?=.*(war|battle))(?=.*(eavy casualt)).*$)
```

## Terrorist Acts

```
(terrorist acts*)
```