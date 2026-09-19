Absolutely — here’s a **detailed Fresh Aqua theme map** built around `#14B8A6`, with enough structure for a website, app, dashboard, or design system. 🎨

## Fresh Aqua — Core Palette

| Token        |           Hex | Suggested use                       |
| ------------ | ------------: | ----------------------------------- |
| Aqua 50      |     `#F0FDFA` | Very light page backgrounds         |
| Aqua 100     |     `#CCFBF1` | Soft sections, selected backgrounds |
| Aqua 200     |     `#99F6E4` | Light highlights, chips             |
| Aqua 300     |     `#5EEAD4` | Decorative accents                  |
| Aqua 400     |     `#2DD4BF` | Hover accents, secondary actions    |
| **Aqua 500** | **`#14B8A6`** | **Primary brand color**             |
| Aqua 600     |     `#0D9488` | Primary hover                       |
| Aqua 700     |     `#0F766E` | Active / pressed state              |
| Aqua 800     |     `#115E59` | Dark brand surfaces                 |
| Aqua 900     |     `#134E4A` | Strong dark contrast                |
| Aqua 950     |     `#042F2E` | Deepest teal / dark mode background |

## Neutral + White Map

A pure white interface works best when paired with slightly tinted neutrals rather than only gray.

| Role                 | Color     |
| -------------------- | --------- |
| Pure White           | `#FFFFFF` |
| Warm White           | `#FCFFFE` |
| Aqua White           | `#F8FFFD` |
| Main Background      | `#F7FBFA` |
| Secondary Background | `#F0F7F6` |
| Subtle Surface       | `#E8F3F1` |
| Border Light         | `#DCEAE8` |
| Border               | `#CBDDD9` |
| Muted Text           | `#647A77` |
| Secondary Text       | `#435C59` |
| Main Text            | `#183B37` |
| Strong Text          | `#102E2B` |
| Near Black           | `#092521` |

---

# UI Color Map

### Page backgrounds

| Element             | Default   | Alternate |
| ------------------- | --------- | --------- |
| Main page           | `#FFFFFF` | `#F8FFFD` |
| App background      | `#F7FBFA` | `#F0FDFA` |
| Section             | `#F8FFFD` | `#F0FDFA` |
| Elevated surface    | `#FFFFFF` | —         |
| Highlighted section | `#CCFBF1` | `#F0FDFA` |
| Dark brand section  | `#134E4A` | `#042F2E` |

---

# Typography

### On light backgrounds

| Role           | Color     |
| -------------- | --------- |
| Heading        | `#102E2B` |
| Body           | `#294B47` |
| Secondary text | `#647A77` |
| Placeholder    | `#8CA5A1` |
| Disabled text  | `#AFC2BF` |
| Link           | `#0D9488` |
| Link hover     | `#0F766E` |

A good hierarchy would be:

**Heading**
`#102E2B`

**Body**
`#294B47`

**Caption**
`#647A77`

This keeps the interface softer than using pure black.

### On aqua/dark backgrounds

| Role           | Color     |
| -------------- | --------- |
| Primary text   | `#FFFFFF` |
| Secondary text | `#CCFBF1` |
| Muted text     | `#99F6E4` |
| Link           | `#5EEAD4` |

---

# Primary Buttons

### Filled button

| State    | Background | Text      |
| -------- | ---------- | --------- |
| Default  | `#14B8A6`  | `#FFFFFF` |
| Hover    | `#0D9488`  | `#FFFFFF` |
| Pressed  | `#0F766E`  | `#FFFFFF` |
| Focus    | `#14B8A6`  | `#FFFFFF` |
| Disabled | `#99F6E4`  | `#F0FDFA` |

Recommended focus ring:

`#5EEAD4`

with roughly **30–40% opacity**.

---

# Secondary Button

| State    | Background | Border    | Text      |
| -------- | ---------- | --------- | --------- |
| Default  | `#FFFFFF`  | `#14B8A6` | `#0D9488` |
| Hover    | `#F0FDFA`  | `#0D9488` | `#0F766E` |
| Pressed  | `#CCFBF1`  | `#0F766E` | `#115E59` |
| Disabled | `#F7FBFA`  | `#DCEAE8` | `#AFC2BF` |

---

# Ghost Button

Default:

* Background: transparent
* Text: `#0D9488`

Hover:

* Background: `#F0FDFA`
* Text: `#0F766E`

Pressed:

* Background: `#CCFBF1`
* Text: `#115E59`

---

# Cards

### Standard card

```text
Background    #FFFFFF
Border        #DCEAE8
Title         #102E2B
Body          #435C59
Muted text    #647A77
Accent        #14B8A6
```

### Highlight card

```text
Background    #F0FDFA
Border        #99F6E4
Title         #115E59
Body          #435C59
Accent        #14B8A6
```

### Strong aqua card

```text
Background    #14B8A6
Title         #FFFFFF
Body          #F0FDFA
Muted         #CCFBF1
```

---

# Navigation

### Light navbar

```text
Background       #FFFFFF
Text             #435C59
Active text      #0F766E
Active indicator #14B8A6
Hover background #F0FDFA
Border           #E8F3F1
```

### Dark navbar

```text
Background       #134E4A
Text             #CCFBF1
Active text      #FFFFFF
Hover background #115E59
Accent           #2DD4BF
```

---

# Sidebar

A particularly clean combination:

```text
Sidebar Background     #042F2E
Primary Text           #FFFFFF
Secondary Text         #99F6E4
Menu Hover             #115E59
Menu Active Background #0F766E
Menu Active Text       #FFFFFF
Icon                   #5EEAD4
Divider                #134E4A
```

This gives Fresh Aqua a more polished **dashboard/SaaS** appearance.

---

# Forms & Inputs

### Default input

```text
Background    #FFFFFF
Text          #183B37
Placeholder   #8CA5A1
Border        #CBDDD9
```

### Hover

```text
Border #99CFC7
```

### Focus

```text
Border     #14B8A6
Focus ring #CCFBF1
```

### Disabled

```text
Background #F0F7F6
Border     #DCEAE8
Text       #AFC2BF
```

---

# Toggle / Checkbox / Radio

### Active

```text
Background #14B8A6
Border     #14B8A6
Icon       #FFFFFF
```

### Hover

```text
Background #0D9488
```

### Inactive

```text
Background #FFFFFF
Border     #9FB8B4
```

---

# Badges / Chips

### Aqua badge

```text
Background #CCFBF1
Text       #0F766E
```

### Strong aqua badge

```text
Background #14B8A6
Text       #FFFFFF
```

### Subtle badge

```text
Background #F0FDFA
Text       #115E59
Border     #99F6E4
```

---

# Table Colors

```text
Table Background   #FFFFFF
Header Background  #F0FDFA
Header Text        #115E59
Row Text           #294B47
Row Divider        #E8F3F1
Row Hover          #F8FFFD
Selected Row       #CCFBF1
```

---

# Semantic Status Colors

You don't want every status to be aqua, because users need immediate visual differentiation.

| Status  | Main      | Light background | Dark text |
| ------- | --------- | ---------------- | --------- |
| Success | `#16A085` | `#E8F8F3`        | `#0B6655` |
| Info    | `#0891B2` | `#E6F8FC`        | `#155E75` |
| Warning | `#F59E0B` | `#FFF7E1`        | `#92400E` |
| Error   | `#E5484D` | `#FFF0F0`        | `#A8242A` |

---

# Charts / Data Visualization

For charts where aqua stays dominant:

| Series   | Color     |
| -------- | --------- |
| Series 1 | `#14B8A6` |
| Series 2 | `#0891B2` |
| Series 3 | `#5EEAD4` |
| Series 4 | `#0F766E` |
| Series 5 | `#7DD3FC` |
| Series 6 | `#83C5BE` |
| Series 7 | `#F59E0B` |
| Series 8 | `#64748B` |

Grid lines:

`#E8F3F1`

Axis text:

`#647A77`

Chart background:

`#FFFFFF`

---

# Gradient Map

### Primary gradient

```css
linear-gradient(135deg, #14B8A6 0%, #0891B2 100%)
```

### Soft aqua gradient

```css
linear-gradient(135deg, #F0FDFA 0%, #CCFBF1 100%)
```

### Deep teal gradient

```css
linear-gradient(135deg, #0F766E 0%, #042F2E 100%)
```

### Hero background

```css
linear-gradient(135deg, #F0FDFA 0%, #FFFFFF 50%, #E6FFFB 100%)
```

---

# Dark Mode Fresh Aqua 🌙

| Role                 | Color     |
| -------------------- | --------- |
| Main Background      | `#071F1D` |
| Secondary Background | `#0B2B28` |
| Surface              | `#103632` |
| Elevated Surface     | `#13413C` |
| Border               | `#20554F` |
| Main Text            | `#F0FDFA` |
| Secondary Text       | `#B6D8D3` |
| Muted Text           | `#7FA8A2` |
| Primary              | `#2DD4BF` |
| Primary Hover        | `#5EEAD4` |
| Primary Active       | `#14B8A6` |
| Dark Accent          | `#0F766E` |

---

# Recommended Overall Combination

For a **modern Fresh Aqua UI**, I'd use this as the default system:

```text
Brand Primary      #14B8A6
Brand Hover        #0D9488
Brand Dark         #0F766E

Page Background    #F7FBFA
Surface            #FFFFFF
Surface Alt        #F0FDFA

Heading            #102E2B
Body               #294B47
Muted Text         #647A77

Border              #DCEAE8
Strong Border       #CBDDD9

Soft Accent         #CCFBF1
Accent              #5EEAD4
Deep Accent         #134E4A
```

### Visual balance

Use roughly **70% white / off-white**, **20% soft aqua neutrals**, and **10% strong teal accents**. That keeps Fresh Aqua feeling **clean, airy, modern, and premium** instead of overwhelmingly turquoise.
