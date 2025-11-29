import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Master Physical AI',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Go beyond digital-only AI. Learn to build embodied intelligence that perceives the physical world, plans actions, and controls humanoid robots using ROS 2, Gazebo, and NVIDIA Isaac.
      </>
    ),
  },
  {
    title: 'Build the "Brains" & "Bodies"',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Bridging the gap between software and hardware. Master the full stack: from high-fidelity simulation and synthetic data generation to deploying Vision-Language-Action (VLA) models on edge compute like NVIDIA Jetson.
      </>
    ),
  },
  {
    title: 'Project-Based Learning',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Apply your knowledge through hands-on modules. Design a robotic nervous system, create a digital twin, train an AI brain, and culminate in a capstone project: building an autonomous humanoid.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
